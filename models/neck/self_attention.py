# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TransT FeatureFusionNetwork class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch.nn.functional as F
import torch
from torch import nn, Tensor


class SelfAttention(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, w):

        src_temp = src_temp.flatten(2).permute(2, 0, 1)

        hs = self.encoder(src1=src_temp)

        hs = hs.permute(1, 0, 2)
        opt = (hs.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, w, w)
        return opt_feat
        # return hs.unsqueeze(0).transpose(1, 2), opt_feat

class Encoder(nn.Module):

    def __init__(self, featurefusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src1):
        # output1 = src1
        # output2 = src2

        for layer in self.layers:
            output1= layer(src1)

        return output1


class FeatureFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm11 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1,
                     src1_mask: Optional[Tensor] = None,
                     src1_key_padding_mask: Optional[Tensor] = None,
                     pos_src1: Optional[Tensor] = None):
        q1 = k1 = self.with_pos_embed(src1, pos_src1)
        src12 = self.self_attn1(q1, k1, value=src1, attn_mask=src1_mask,
                               key_padding_mask=src1_key_padding_mask)[0]
        src1 = src1 + self.dropout11(src12)
        src1 = self.norm11(src1)

        return src1

    def forward(self, src1,
                src1_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None):

        return self.forward_post(src1, src1_mask,
                                 src1_key_padding_mask, pos_src1)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
