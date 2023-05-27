# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Revised by Jilai Zheng, for USOT
# ------------------------------------------------------------------------------
from __future__ import absolute_import
import os
import cv2
import random
import argparse
import numpy as np
import importlib
import sys
#sys.path.insert(0, '../lib')
sys.path.append("..")
from os.path import exists, join, dirname, realpath, abspath

import lib.models.models as models

from lib.tracker.usot_tracker import USOTTracker
from easydict import EasyDict as edict
from lib.utils.train_utils import load_pretrain_test
from lib.utils.test_utils import cxy_wh_2_rect, get_axis_aligned_bbox, poly_iou
from lib.dataset_loader.benchmark import load_dataset


def parse_args():
    """
    args for USOT testing.
    """
    parser = argparse.ArgumentParser(description='USOT testing')
    parser.add_argument('--arch', dest='arch', default='USOT', help='backbone architecture')
    parser.add_argument('--resume', default='/home/cscv/Documents/lsl/USOTFormer/scripts/VOT_train/TFF-Corr-motion/checkpoint_e30.pth', type=str, help='pretrained model')
    parser.add_argument('--dataset', default='GTOT', choices={'GTOT', 'RGB-T234', 'LasHeR'}, help='dataset test')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    parser.add_argument('--version', default='v1', help='testing style version')
    #parser.add_argument('--Feature_Backbone', type=str, choices=['ResNet', 'Vit'], default='Vit')
    args = parser.parse_args()

    return args

speed = []
def track(tracker, net, video, args):
    start_frame, toc = 0, 0
    visualization = True

    # Save result to evaluate
    if args.epoch_test:
        suffix = args.resume.split('/')[-1]
        suffix = suffix.split('.')[0]
        tracker_path = os.path.join('var/result', args.dataset, 'test')
    elif 'LasHeR' in args.dataset:
        tracker_path = os.path.join('var/result', args.dataset, 'test_tracking_result')
    else:
        tracker_path = os.path.join('var/result', args.dataset, 'test')

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in args.dataset:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    elif 'GOT' in args.dataset:
        video_path = os.path.join(tracker_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    elif 'LasHeR' in args.dataset:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))
    else:
        result_path = os.path.join(tracker_path, 'test_' + '{:s}.txt'.format(video['name']))

    if os.path.exists(result_path):
        return

    regions = []
    image_files_color, image_files_ir, gt = video['image_files_color'], video['image_files_ir'], video['gt']
    n_images = len(image_files_color)
    #for f, image_file in enumerate(image_files):
    for f in range(0, n_images):

        im_color = cv2.imread(image_files_color[f])
        im_ir = cv2.imread(image_files_ir[f])
        if len(im_color.shape) == 2:
            # Align with training
            im_color = cv2.cvtColor(im_color, cv2.COLOR_GRAY2BGR)
            im_ir = cv2.cvtColor(im_ir, cv2.COLOR_GRAY2BGR)

        tic = cv2.getTickCount()

        # Init procedure
        if f == start_frame:
            cx, cy, w, h = get_axis_aligned_bbox(gt[f], args.dataset)
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            # Init tracker
            state = tracker.init(im_color, im_ir, target_pos, target_sz, net)
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            if args.dataset == 'GTOT':
                gt[f][2] = gt[f][2] - gt[f][0]
                gt[f][3] = gt[f][3] - gt[f][1]
            regions.append(gt[f])

        # Tracking procedure
        elif f > start_frame:
            state = tracker.track(state, im_color, im_ir)
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            b_overlap = poly_iou(gt[f], location) if 'VOT' in args.dataset else 1
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append(2)
                start_frame = f + 5
        else:
            regions.append(0)

        toc += cv2.getTickCount() - tic

        if visualization:
            im_show = cv2.cvtColor(im_color, cv2.COLOR_RGB2BGR)
            cv2.rectangle(im_show, (int(state['target_pos'][0] - (state['target_sz'][0])/2), int(state['target_pos'][1] - (state['target_sz'][1])/2)),
                          (int(state['target_pos'][0] + (state['target_sz'][0])/2), int(state['target_pos'][1] + (state['target_sz'][1])/2)),
                          (0, 255, 0), 3)
            cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('img', im_show)
            cv2.waitKey(1)

    with open(result_path, "w") as fin:
        if 'VOT' in args.dataset:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')
        else:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    toc /= cv2.getTickFrequency()
    fps = n_images / toc
    speed.append(fps)
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(video['name'], toc, f / toc))


def main():
    args = parse_args()

    # Prepare model
    net = models.__dict__[args.arch]()
    net = load_pretrain_test(net, args.resume)
    net.eval()
    net = net.cuda()

    # Prepare video
    dataset = load_dataset(args.dataset)
    video_keys = list(dataset.keys()).copy()

    # Prepare tracker
    info = edict()
    info.arch = args.arch
    info.dataset = args.dataset
    info.epoch_test = args.epoch_test
    info.version = args.version

    if info.arch == 'USOT':
        tracker = USOTTracker(info)
    else:
        assert False, "Warning: Model should be USOT, but currently {}.".format(info.arch)

    # Tracking all videos in benchmark
    for video in video_keys:
        track(tracker, net, dataset[video], args)
    print('***Total Mean Speed: {:3.1f} (FPS)***'.format(np.mean(speed)))


if __name__ == '__main__':
    main()
