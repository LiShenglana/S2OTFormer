# -----------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Revised by Jilai Zheng, for USOT
# ------------------------------------------------------------------------------
import cv2
import os
from lib.vis.visdom_cus import Visdom
from lib.models.connect import AdjustLayer, box_tower_reg, Project
from lib.models.backbones import ResNet50
from lib.models.prroi_pool.functional import prroi_pool2d
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import importlib
import torch.nn.functional as F

from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)
from models.neck.featurefusion_network import FeatureFusionNetwork
from models.neck.self_attention import SelfAttention
from models.neck.correlation import Correlation
from PIL import Image
import time
from torchvision import transforms
import visdom


class USOT_(nn.Module):
    def __init__(self, mem_size=4, pr_pool=True,
                 search_size=255, score_size=25, maximum_batch=16, sf_size=25):
        super(USOT_, self).__init__()
        self.fuse = None
        self.backbone_net_RGB = None
        self.backbone_net_T = None
        self.connect_model = None
        self.zf = None
        self.zf_att = None
        self.zf_color = None
        self.zf_ir = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.neck = None
        self.project = None
        self.search_size = search_size
        self.score_size = score_size
        self.search_feature_size = sf_size
        self.input_proj = None
        self.input_proj1 = None
        self.featurefusion_network = None
        self.selfattention_network = None
        self.correlation = None
        self.class_embed = None
        self.bbox_embed = None
        self.debug = True
        self.use_visdom = False
        self.modality = 'RGB-T'  # RGB T RGB-T
        self.fuse_method = 'Cross_Attention' # Add Cross_Attention
        if self.debug:
            if not self.use_visdom:
                self.save_dir = 'debug'
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

        # Param maximum batch is only used for generating a valid grid, and does not effect the actual batch size
        self.maximum_batch = maximum_batch if self.training else 1
        # Number of memory frames
        self.mem_size = mem_size

        # Always keep pr_pool = True for training a bbox regression module
        self.pr_pool = pr_pool

        self.grids()

    def _init_visdom(self, visdom_info, debug):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        self.next_seq = False
        if debug > 0 and visdom_info.get('use_visdom', True):
            try:
                self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                                     visdom_info=visdom_info)
                self.vis = visdom.Visdom()

                # # Show help
                # help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                #             'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                #             'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                #             'block list.'
                # self.visdom.register(help_text, 'text', 1, 'Help')
            except:
                time.sleep(0.5)
                print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                      '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True

            elif data['key'] == 'n':
                self.next_seq = True

    # def feature_extractor_ResNet_RGB(self, x):
    #     return self.features_RGB(x)
    #
    # def feature_extractor_ResNet_T(self, x):
    #     return self.features_T(x)

    def feature_extractor(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                temp=True):
        self.fuse.finetune_track(cfg=self.cfg, patch_start_index=1)
        x, aux_dict = self.fuse(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, temp=temp)
        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]

        if temp is True:
            enc_opt = feat_last[:, -self.feat_len_z:]  # encoder output for the search region (B, HW, C)
            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_z, self.feat_sz_z)
        else:
            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        return opt_feat

    def _cls_loss(self, pred, label, select):
        if len(select.size()) == 0:
            return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)

    def _weighted_BCE(self, pred, label):
        pred = pred.view(-1)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze().cuda()
        neg = label.data.eq(0).nonzero().squeeze().cuda()
        # pos = label.data.eq(1).nonzero( as_tuple=False).squeeze().cuda()
        # neg = label.data.eq(0).nonzero( as_tuple=False).squeeze().cuda()

        loss_pos = self._cls_loss(pred, label, pos)
        loss_neg = self._cls_loss(pred, label, neg)

        return loss_pos * 0.5 + loss_neg * 0.5

    def _IOULoss(self, pred, target):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        assert losses.numel() != 0
        return losses.mean()

    def add_iouloss(self, bbox_pred, reg_target, reg_weight):
        """
        Add IoU Loss for bbox regression
        """

        bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_target_flatten = reg_target.reshape(-1, 4)
        reg_weight_flatten = reg_weight.reshape(-1)
        pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1)

        bbox_pred_flatten = bbox_pred_flatten[pos_inds]
        reg_target_flatten = reg_target_flatten[pos_inds]

        loss = self._IOULoss(bbox_pred_flatten, reg_target_flatten)

        return loss

    def grids(self):
        """
        Each element of feature map on response map
        :return: H*W*2 (position for each element)
        """
        # Grid for response map
        sz = self.score_size
        stride = 8
        sz_x = sz // 2
        sz_y = sz // 2
        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * stride + self.search_size // 2
        self.grid_to_search_y = y * stride + self.search_size // 2

        self.grid_to_search_x = torch.Tensor(self.grid_to_search_x).unsqueeze(0).unsqueeze(0).cuda()
        self.grid_to_search_y = torch.Tensor(self.grid_to_search_y).unsqueeze(0).unsqueeze(0).cuda()

        self.grid_to_search_x = self.grid_to_search_x.repeat(self.maximum_batch * self.mem_size, 1, 1, 1)
        self.grid_to_search_y = self.grid_to_search_y.repeat(self.maximum_batch * self.mem_size, 1, 1, 1)

        # Axis for search area feature
        sz = self.search_feature_size
        stride = 8
        sz_x = sz // 2
        self.search_area_x_axis = (np.arange(0, sz) - np.floor(float(sz_x))) * stride + self.search_size // 2

    def pred_offset_to_image_bbox(self, bbox_pred, batch):
        """
        Convert bbox from the predicted response map axis to the image-level axis
        """

        self.grid_to_search_x = self.grid_to_search_x[0:batch * self.mem_size].to(bbox_pred.device)
        self.grid_to_search_y = self.grid_to_search_y[0:batch * self.mem_size].to(bbox_pred.device)

        pred_x1 = self.grid_to_search_x - bbox_pred[:, 0, ...].unsqueeze(1)
        pred_y1 = self.grid_to_search_y - bbox_pred[:, 1, ...].unsqueeze(1)
        pred_x2 = self.grid_to_search_x + bbox_pred[:, 2, ...].unsqueeze(1)
        pred_y2 = self.grid_to_search_y + bbox_pred[:, 3, ...].unsqueeze(1)

        pred = [pred_x1, pred_y1, pred_x2, pred_y2]

        pred = torch.cat(pred, dim=1)

        return pred

    def image_bbox_to_prpool_bbox(self, image_bbox):
        """
        Convert bbox from the image axis to the search area axis
        """

        reg_min = self.search_area_x_axis[0]
        reg_max = self.search_area_x_axis[-1]
        sz = 2 * (self.search_feature_size // 2)
        gap = (reg_max - reg_min) / sz
        image_bbox = torch.clamp(image_bbox, max=reg_max + 2 * gap, min=reg_min - 2 * gap)

        slope = 1.0 / gap
        return (image_bbox - reg_min) * slope

    def prpool_feature(self, features, bboxs):
        """
        PrPool from deep features according to specific bboxes
        """

        batch_index = torch.arange(0, features.shape[0]).view(-1, 1).float().to(features.device)
        bboxs_index = torch.cat((batch_index, bboxs), dim=1)
        return prroi_pool2d(features, bboxs_index, 7, 7, 1.0)

    def template(self, z_color, z_ir, template_bbox=None):

        if self.modality == 'RGB-T':
            _, zf_color = self.backbone_net_RGB(z_color)
            _, zf_ir = self.backbone_net_T(z_ir)
            # PrPool the template feature and down-sample the deep features
            # if self.neck is not None:
            #     _, zf_color = self.neck(zf_color, crop=True, pr_pool=self.pr_pool, bbox=template_bbox)
            #     _, zf_ir = self.neck(zf_ir, crop=True, pr_pool=self.pr_pool, bbox=template_bbox)
            if self.fuse_method == 'Add':
                self.zf = torch.add(zf_color, zf_ir)
            elif self.fuse_method == 'Cross_Attention':
                _, _, _, Wz = zf_color.shape
                zf_cat = torch.cat((zf_color, zf_ir), 1)
                self.zf_att = self.featurefusion_network(self.input_proj1(zf_cat), self.input_proj2(zf_color),
                                                    self.input_proj2(zf_ir), Wz)
                zf = torch.add(self.input_proj2(zf_color), self.input_proj2(zf_ir))
                self.zf = torch.add(self.zf_att, zf)
        elif self.modality == 'RGB':
            _, zf_color = self.backbone_net_RGB(z_color)
            self.zf = zf_color
            # PrPool the template feature and down-sample the deep features
            # if self.neck is not None:
            #     _, self.zf = self.neck(zf_color, crop=True, pr_pool=self.pr_pool, bbox=template_bbox)
        elif self.modality == 'T':
            _, zf_ir = self.backbone_net_T(z_ir)
            self.zf = zf_ir
            # PrPool the template feature and down-sample the deep features
            # if self.neck is not None:
            #     _, self.zf = self.neck(zf_ir, crop=True, pr_pool=self.pr_pool, bbox=template_bbox)
        # elif self.cfg.MODEL.BACKBONE.Feature_Backbone == 'Vit':
        #     self.zf = self.feature_extractor(z_ir, z_color, temp=True)
        #     self.zf = self.project(self.zf)
        #     # _, self.zf = self.feature_extractor_ResNet(self.zf)
        # # self.zf = zf_color

        # feature_map = zf_color
        # # feature_map = feature_map.permute(0, 2, 3, 1)
        # feature_map = feature_map.cpu().detach().numpy()
        # pathfea = '/home/cscv/Documents/lsl/USOT/scripts/feature_map_save/Feature_map_save/'
        # for index in range(feature_map.shape[0]):
        #     feature_map_i = feature_map[index]
        #     for j in range(feature_map_i.shape[2]):
        #         image = Image.fromarray(np.uint8(feature_map_i[j]))
        #         timestamp = datetime.datetime.now().strftime("%M-%S")
        #         savepath = pathfea + timestamp + '_r.jpg'
        #         image.save(savepath)

        if self.neck is not None:
            _, self.zf = self.neck(self.zf, crop=True, pr_pool=self.pr_pool, bbox=template_bbox)

    def track(self, x_color, x_ir, template_mem=None, score_mem=None):

        if self.modality == 'RGB-T':
            _, xf_color = self.backbone_net_RGB(x_color)
            _, xf_ir = self.backbone_net_T(x_ir)
            # PrPool the template feature and down-sample the deep features
            # if self.neck is not None:
            #     xf_color = self.neck(xf_color)
            #     xf_ir = self.neck(xf_ir)
            if self.fuse_method == 'Add':
                xf = torch.add(xf_color, xf_ir)
            elif self.fuse_method == 'Cross_Attention':
                _, _, _, Wx = xf_color.shape
                xf_cat = torch.cat((xf_color, xf_ir), 1)
                xf_att = self.featurefusion_network(self.input_proj1(xf_cat), self.input_proj2(xf_color),
                                                    self.input_proj2(xf_ir), Wx)
                xf = torch.add(self.input_proj2(xf_color), self.input_proj2(xf_ir))
                xf = torch.add(xf_att, xf)
        elif self.modality == 'RGB':
            _, xf_color = self.backbone_net_RGB(x_color)
            xf = xf_color
            # PrPool the template feature and down-sample the deep features
            # if self.neck is not None:
            #     xf = self.neck(xf_color)
        elif self.modality == 'T':
            _, xf_ir = self.backbone_net_T(x_ir)
            xf = xf_ir
            # PrPool the template feature and down-sample the deep features
            # if self.neck is not None:
            #     xf = self.neck(xf_ir)

        # elif self.cfg.MODEL.BACKBONE.Feature_Backbone == 'Vit':
        #     xf = self.feature_extractor(x_ir, x_color, temp=False)
        #     xf = self.project(xf)
        #     # _, xf = self.feature_extractor_ResNet(xf)
        # xf = xf_color

        if self.neck is not None:
            xf = self.neck(xf)

        if template_mem is not None:

            # _, _, _, W = template_mem.shape
            # template_mem = self.selfattention_network(template_mem, W)
            # Track with both offline and online module (with memory queue features existing)
            bbox_pred, cls_pred, cls_feature, reg_feature, cls_memory_pred = self.connect_model(xf, kernel=self.zf,
                                                                                                memory_kernel=template_mem,
                                                                                                memory_confidence=score_mem)
            corr = self.correlation(xf_att, self.zf_att)
            corr = self.class_embed(corr)
            corr = self.change(corr, xf_att.shape[3])
            # cls_pred = cls_pred.squeeze(0)
            # c, h, w = cls_pred.size()
            # mask = torch.zeros((c, h, w)).cuda()
            # cls_pred = torch.where(cls_pred > 0, cls_pred, mask)
            # r = 3
            # for j in range(len(cls_pred)):
            #     ind = torch.nonzero(cls_pred[j] == torch.max(cls_pred[j]))[0]
            #     mask[j][ind[0]-r:ind[0]+r, ind[1]-r:ind[1]+r] = 1
            # cls_pred = cls_pred * mask
            # cls_pred = cls_pred.unsqueeze(0)

            if self.debug:
                if self.use_visdom:
                    self.visdom.register(np.squeeze(x_color, 0), 'image', 1, 'search_color')
                    self.visdom.register(np.squeeze(x_ir, 0), 'image', 1, 'search_ir')
                    self.visdom.register(cls_pred.view(cls_memory_pred.shape[2], cls_memory_pred.shape[3]), 'heatmap', 1, 'cls_memory')
                    self.visdom.register(cls_pred.view(cls_pred.shape[2], cls_pred.shape[3]), 'heatmap', 1, 'score_map')
                    self.visdom.register(corr.view(corr.shape[2], corr.shape[3]), 'heatmap', 1, 'corr')

                    while self.pause_mode:
                        if self.step:
                            self.step = False
                            break

            # Here xf is the feature of search areas which will be cropped soon according to the final bbox
            # return cls_pred, bbox_pred, cls_memory_pred, xf, corr
            return cls_pred, bbox_pred, cls_memory_pred, xf
        else:
            # Track with offline module only
            bbox_pred, cls_pred, _, _, _ = self.connect_model(xf, kernel=self.zf)

            return cls_pred, bbox_pred, None, None

    def extract_memory_feature_ResNet_fuse(self, ori_x_color=None, ori_x_ir=None, xf=None, search_bbox=None):
        # Note that here search bbox is the bbox on the deep feature (not on the original search frame)
        if ori_x_color is not None:
            # _, xf = self.feature_extractor_ResNet(ori_x)
            _, xf_color = self.backbone_net_RGB(ori_x_color)
            # xf = self.neck(xf, crop=False)
            _, xf_ir = self.backbone_net_T(ori_x_ir)
            if self.fuse_method == 'Add':
                xf = torch.add(xf_color, xf_ir)
            elif self.fuse_method == 'Cross_Attention':
                _, _, _, Wx = xf_color.shape
                xf_cat = torch.cat((xf_color, xf_ir), 1)
                xf_att = self.featurefusion_network(self.input_proj1(xf_cat), self.input_proj2(xf_color),
                                                    self.input_proj2(xf_ir), Wx)
                xf = torch.add(self.input_proj2(xf_color), self.input_proj2(xf_ir))
                xf = torch.add(xf_att, xf)
            xf = self.neck(xf, crop=False)
        features = self.prpool_feature(xf, search_bbox)
        return features

    def extract_memory_feature_ResNet(self, ori_x=None, xf=None, search_bbox=None, img='RGB'):
        # Note that here search bbox is the bbox on the deep feature (not on the original search frame)
        if ori_x is not None:
            # _, xf = self.feature_extractor_ResNet(ori_x)
            if img is 'RGB':
                _, xf = self.backbone_net_RGB(ori_x)
                xf = self.neck(xf, crop=False)
            elif img is 'T':
                _, xf = self.backbone_net_T(ori_x)
                xf = self.neck(xf, crop=False)
        features = self.prpool_feature(xf, search_bbox)
        return features


    # def extract_memory_feature(self, ori_x_ir=None, ori_x_color=None, xf=None, search_bbox=None, temp=False):
    #     # Note that here search bbox is the bbox on the deep feature (not on the original search frame)
    #     if ori_x_ir is not None:
    #         xf = self.feature_extractor(ori_x_ir, ori_x_color, temp=temp)
    #         xf = self.project(xf)
    #         _, xf = self.feature_extractor_ResNet(xf)
    #         xf = self.neck(xf, crop=False)
    #     features = self.prpool_feature(xf, search_bbox)
    #     return features

    def forward(self, template_color, search_color, template_ir, search_ir, label=None, reg_target=None,
                reg_weight=None, template_bbox=None, search_memory_color=None, search_memory_ir=None,
                search_bbox=None, cls_ratio=0.40, label2=None):
        """
        Training pipeline for both naive Siamese and cycle memory
        """
        # Feature extraction for template patch and search area
        if self.modality == 'RGB-T':
            _, zf_color = self.backbone_net_RGB(template_color)
            _, xf_color = self.backbone_net_RGB(search_color)
            _, zf_ir = self.backbone_net_T(template_ir)
            _, xf_ir = self.backbone_net_T(search_ir)
            if self.fuse_method == 'Add':
                zf = torch.add(zf_color, zf_ir)
                xf = torch.add(xf_color, xf_ir)
            elif self.fuse_method == 'Cross_Attention':
                _, _, _, Wz = zf_color.shape
                _, _, _, Wx = xf_color.shape
                # zf = self.featurefusion_network(zf_ir, zf_color, Wz)
                # xf = self.featurefusion_network(xf_ir, xf_color, Wx)
                zf_cat = torch.cat((zf_color, zf_ir), 1)
                xf_cat = torch.cat((xf_color, xf_ir), 1)
                zf_att = self.featurefusion_network(self.input_proj1(zf_cat), self.input_proj2(zf_color), self.input_proj2(zf_ir), Wz)
                xf_att = self.featurefusion_network(self.input_proj1(xf_cat), self.input_proj2(xf_color), self.input_proj2(xf_ir), Wx)
                zf = torch.add(self.input_proj2(zf_color), self.input_proj2(zf_ir))
                zf = torch.add(zf_att, zf)
                xf = torch.add(self.input_proj2(xf_color), self.input_proj2(xf_ir))
                xf = torch.add(xf_att, xf)
        elif self.modality == 'RGB':
            _, zf_color = self.backbone_net_RGB(template_color)
            _, xf_color = self.backbone_net_RGB(search_color)
            zf = zf_color
            xf = xf_color
        elif self.modality == 'T':
            _, zf_ir = self.backbone_net_T(template_ir)
            _, xf_ir = self.backbone_net_T(search_ir)
            zf = zf_ir
            xf = xf_ir

        #PrPool the template feature and down-sample the deep features
        if self.neck is not None:
            _, zf = self.neck(zf, crop=True, pr_pool=self.pr_pool, bbox=template_bbox)
            xf = self.neck(xf, crop=False)

        if search_memory_color is not None:
            # Original Siamese fg/bg cls and bbox reg branch (self-track in paper)
            bbox_pred, cls_pred, cls_x, _, _ = self.connect_model(xf, kernel=zf)
            # corr = self.correlation(xf_att, zf_att)
            # corr = self.change(self.class_embed(corr), w=31)

            # Add bbox regression loss
            reg_loss = self.add_iouloss(bbox_pred, reg_target, reg_weight)
            # Add naive Siamese cls loss
            cls_loss_ori = self._weighted_BCE(cls_pred, label)
            # correlation_loss = self._weighted_BCE(corr, label2)

            # Now begin to calculate cycle memory loss
            # Extract deep features for memory search areas
            batch, mem_size, cx, hx, wx = search_memory_color.shape
            search_memory_color = search_memory_color.view(-1, cx, hx, wx)
            search_memory_ir = search_memory_ir.view(-1, cx, hx, wx)
            if self.modality == 'RGB-T':
                _, xf_mem_color = self.backbone_net_RGB(search_memory_color)
                _, xf_mem_ir = self.backbone_net_T(search_memory_ir)
                if self.fuse_method == 'Add':
                    xf_mem = torch.add(xf_mem_color, xf_mem_ir)
                elif self.fuse_method == 'Cross_Attention':
                    _, _, _, Wz = xf_mem_color.shape
                    xf_mem_cat = torch.cat((xf_mem_color, xf_mem_ir), 1)
                    xf_mem_att = self.featurefusion_network(self.input_proj1(xf_mem_cat), self.input_proj2(xf_mem_color), self.input_proj2(xf_mem_ir), Wx)
                    xf_mem = torch.add(self.input_proj2(xf_mem_color), self.input_proj2(xf_mem_ir))
                    xf_mem = torch.add(xf_mem_att, xf_mem)
            elif self.modality == 'RGB':
                _, xf_mem_color = self.backbone_net_RGB(search_memory_color)
                xf_mem = xf_mem_color
                # xf_mem = self.neck(xf_mem_color, crop=False)
            elif self.modality == 'T':
                _, xf_mem_ir = self.backbone_net_T(search_memory_ir)
                xf_mem = xf_mem_ir

            xf_mem = self.neck(xf_mem, crop=False)
            # Prepare feature for mem_forward_cls (forward tracking with the online module)
            search_pooled_feature = self.prpool_feature(xf, search_bbox)
            batch, cspf, wspf, hspf = search_pooled_feature.shape
            search_pooled_feature = search_pooled_feature.view(batch, 1, cspf, wspf, hspf)
            search_pooled_feature = search_pooled_feature.repeat(1, mem_size, 1, 1, 1)
            search_pooled_feature = search_pooled_feature.view(-1, cspf, wspf, hspf)

            batch, cspf, wspf, hspf = xf_att.shape
            search_feature_att = xf_att.view(batch, 1, cspf, wspf, hspf)
            search_feature_att = search_feature_att.repeat(1, mem_size, 1, 1, 1)
            search_feature_att = search_feature_att.view(-1, cspf, wspf, hspf)

            # Repeat the original template
            batch, cz, hz, wz = zf.shape
            zf_mem = zf.view(batch, 1, cz, hz, wz)
            zf_mem = zf_mem.repeat(1, mem_size, 1, 1, 1)
            zf_mem = zf_mem.view(-1, cz, hz, wz)

            batch, cz, hz, wz = zf_att.shape
            zf_mem_att = zf_att.view(batch, 1, cz, hz, wz)
            zf_mem_att = zf_mem_att.repeat(1, mem_size, 1, 1, 1)
            zf_mem_att = zf_mem_att.view(-1, cz, hz, wz)

            # Get the intermediate target bbox and cls score in memory search areas (tracking with offline module)
            off_forward_bbox, off_forward_cls, forward_x_store, _, _ = self.connect_model(xf_mem, kernel=zf_mem)

            corr_off = self.correlation(xf_mem_att, zf_mem_att)
            corr_off = self.change(self.class_embed(corr_off), w=31)
            # correlation_loss_off = self._weighted_BCE(corr_off, label2)

            # Get the mem_forward_cls score in memory search areas (tracking with online module)
            fake_confidence = torch.ones(batch * mem_size, 1)
            _, _, _, _, mem_forward_cls = self.connect_model(xf_mem, memory_kernel=search_pooled_feature,
                                                             memory_confidence=fake_confidence,
                                                             cls_x_store=forward_x_store)
            corr_mem = self.correlation(xf_mem_att, search_feature_att)
            corr_mem = self.change(self.class_embed(corr_mem), w=31)
            Resize_to = transforms.Resize([mem_forward_cls.shape[2], mem_forward_cls.shape[3]])
            corr_mem = Resize_to(corr_mem)
            corr_off = Resize_to(corr_off)

            mem_forward_cls = mem_forward_cls.view(batch, mem_size, -1)
            off_forward_cls = off_forward_cls.view(batch, mem_size, -1)
            corr_mem = corr_mem.view(batch, mem_size, -1)
            corr_off = corr_off.view(batch, mem_size, -1)

            # Linearly combine off_forward_cls and mem_forward_cls as the forward response map
            # Note: weighted add memory_forward_cls and off_forward_cls, while bbox remains
            forward_res_map = cls_ratio * (off_forward_cls + corr_off) + (1 - cls_ratio) * (mem_forward_cls + corr_mem)
            # forward_res_map = cls_ratio * off_forward_cls + (1 - cls_ratio) * mem_forward_cls
            best_forward_cls = forward_res_map.max(dim=2)
            best_forward_cls_argmax = best_forward_cls.indices.view(batch, mem_size, 1, 1)
            best_forward_cls_argmax = best_forward_cls_argmax.repeat(1, 1, 1, 4)
            bbox_pred_to_img = self.pred_offset_to_image_bbox(off_forward_bbox, batch)
            bbox_pred_to_img = bbox_pred_to_img.view(batch, mem_size, 4, -1).transpose(2, 3)
            best_mem_bbox = torch.gather(bbox_pred_to_img, dim=2,
                                         index=best_forward_cls_argmax).view(batch * mem_size, 4)
            best_forward_cls_score = best_forward_cls.values.detach()
            best_forward_bbox_pool = self.image_bbox_to_prpool_bbox(best_mem_bbox).detach()

            # PrPool intermediate features from memory search areas as the memory queue
            pooled_mem_features = self.prpool_feature(xf_mem, best_forward_bbox_pool) #[48, 256, 7, 7]

            #self-attentin between memory features as time attention
            # _, _, _, W = pooled_mem_features.shape
            # pooled_mem_features_att = self.selfattention_network(pooled_mem_features, W)
            # Backward track from memory queue to the search area in the template frame
            _, _, _, _, backward_res_map = self.connect_model(xf, memory_kernel=pooled_mem_features,
                                                              memory_confidence=best_forward_cls_score,
                                                              cls_x_store=cls_x)  #[6, 1, 25, 25] xf_mem[24,256,31,31]
            if self.debug:
                if self.use_visdom:
                    for index in range(label.shape[0]):
                        self.visdom.register(backward_res_map[index].view(backward_res_map.shape[2], backward_res_map.shape[3]), 'heatmap', 1, 'backward_res_map')
                        self.visdom.register(cls_pred[index].view(25, 25), 'heatmap', 1, 'score_map_train')
                        # self.visdom.register(corr[index].view(31, 31), 'heatmap', 1, 'corr')
                        self.visdom.register(label[index].view(25, 25), 'heatmap', 1, 'label_map_train')
                        self.visdom.register(np.squeeze(search_memory_color[index], 0), 'image', 1,
                                             'search_memory_color')
                        self.visdom.register(np.squeeze(search_memory_ir[index], 0), 'image', 1, 'search_memory_ir')
                        self.visdom.register(np.squeeze(template_color[index], 0), 'image', 1, 'template_color')
                        self.visdom.register(np.squeeze(template_ir[index], 0), 'image', 1, 'template_ir')
                        self.visdom.register(np.squeeze(search_color[index], 0), 'image', 1, 'search_color')
                        self.visdom.register(np.squeeze(search_ir[index], 0), 'image', 1, 'search_ir')

                    while self.pause_mode:
                        if self.step:
                            self.step = False
                            break

            # Cycle memory loss is calculated with the same pseudo label as original cls loss
            cls_memory_loss = self._weighted_BCE(backward_res_map, label)
            torch.cuda.empty_cache()

            return cls_loss_ori, cls_memory_loss, reg_loss

        else:
            # The following logic is for purely offline naive Siamese training
            bbox_pred, cls_pred, _, _, _ = self.connect_model(xf, kernel=zf)
            # corr = self.correlation(xf_att, zf_att)
            # corr = self.change(self.class_embed(corr), w=31)
            # correlation_loss = self._weighted_BCE(corr, label2)
            # empty_weight = torch.ones(2)
            # empty_weight[-1] = 0.0625
            # correlation_loss = F.cross_entropy(corr.transpose(1, 2), label2, self.empty_weight)

            # for index in range(cls_pred.shape[0]):
            #     c, h, w = cls_pred[index].size()
            #     mask = torch.zeros((c, h, w)).cuda()
            #     cls_pred[index] = torch.where(cls_pred[index] > 0, cls_pred[index], mask)
            #     r = 3
            #     for j in range(len(cls_pred[index])):
            #         ind = torch.nonzero(cls_pred[index][j] == torch.max(cls_pred[index][j]))[0]
            #         mask[j][ind[0] - r:ind[0] + r, ind[1] - r:ind[1] + r] = 1
            #     cls_pred[index] = cls_pred[index] * mask

            #print('cls:', cls_pred)
            #print('label:', label)
            if self.debug:
                if self.use_visdom:
                    for index in range(cls_pred.shape[0]):
                        self.visdom.register(cls_pred[index].view(25, 25), 'heatmap', 1, 'score_map_train')
                        # self.visdom.register(corr[index].view(31, 31), 'heatmap', 1, 'corr')
                        self.visdom.register(label[index].view(25, 25), 'heatmap', 1, 'label_map_train')
                        self.visdom.register(np.squeeze(template_color[index], 0), 'image', 1, 'template_color')
                        self.visdom.register(np.squeeze(search_color[index], 0), 'image', 1, 'search_color')
                        self.visdom.register(np.squeeze(template_ir[index], 0), 'image', 1, 'template_ir')
                        self.visdom.register(np.squeeze(search_ir[index], 0), 'image', 1, 'search_ir')

                    while self.pause_mode:
                        if self.step:
                            self.step = False
                            break

            cls_loss = self._weighted_BCE(cls_pred, label)
            reg_loss = self.add_iouloss(bbox_pred, reg_target, reg_weight)
            torch.cuda.empty_cache()

            return cls_loss, None, reg_loss
            # return cls_loss, None, reg_loss, correlation_loss

    def change(self, X=None, w=63):
        opt = (X.permute((1, 3, 0, 2)).contiguous())
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, w, w)
        return opt_feat

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class USOT(USOT_):
    def __init__(self, settings=None):
        if settings is None:
            settings = {'mem_size': 4, 'pr_pool': True}
        super(USOT, self).__init__(mem_size=settings['mem_size'], pr_pool=settings['pr_pool'],
                                   search_size=255, score_size=25, maximum_batch=16, sf_size=25)
        if self.modality == 'RGB-T':
            self.backbone_net_RGB = ResNet50(used_layers=[3])  # in param
            self.backbone_net_T = ResNet50(used_layers=[3])  # in param
            if self.fuse_method == 'Cross_Attention':
                self.featurefusion_network = FeatureFusionNetwork(
                    d_model=256,
                    dropout=0.1,
                    nhead=8,
                    dim_feedforward=2048,
                    num_featurefusion_layers=1
                )
                # self.correlation = Correlation(
                #     d_model=256,
                #     dropout=0.1,
                #     nhead=8,
                #     dim_feedforward=2048,
                #     num_featurefusion_layers=1
                # )
                # self.selfattention_network = SelfAttention(
                #     d_model=256,
                #     dropout=0.1,
                #     nhead=8,
                #     dim_feedforward=2048,
                #     num_featurefusion_layers=1
                # )
                self.input_proj1 = nn.Conv2d(2048, 256, kernel_size=1)
                self.input_proj2 = nn.Conv2d(1024, 256, kernel_size=1)
                self.neck = AdjustLayer(in_channels=256, out_channels=256, pr_pool=settings['pr_pool'])
                self.class_embed = MLP(256, 256, 1, 3)
            elif self.fuse_method == 'Add':
                self.neck = AdjustLayer(in_channels=1024, out_channels=256, pr_pool=settings['pr_pool'])
        elif self.modality == 'RGB':
            self.backbone_net_RGB = ResNet50(used_layers=[3])  # in param
            self.neck = AdjustLayer(in_channels=1024, out_channels=256, pr_pool=settings['pr_pool'])
        elif self.modality == 'T':
            self.backbone_net_T = ResNet50(used_layers=[3])  # in param
            self.neck = AdjustLayer(in_channels=1024, out_channels=256, pr_pool=settings['pr_pool'])
        self.connect_model = box_tower_reg(in_channels=256, out_channels=256, tower_num=4)