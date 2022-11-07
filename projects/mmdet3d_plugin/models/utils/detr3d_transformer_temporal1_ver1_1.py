import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import math
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE,
                                      )
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from projects.mmdet3d_plugin.models.utils.detr3d_transformer import feature_sampling, inverse_sigmoid
# def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
# def inverse_sigmoid(x, eps=1e-5):
@ATTENTION.register_module()
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, start_pos=0) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[start_pos:start_pos+x.size(0)]
        return x
        # return self.dropout(x)

@ATTENTION.register_module()
class Detr3DCrossAtten_T1v1_1(BaseModule):
    """ add temporal embeddings for queries"""
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DCrossAtten_T1v1_1, self).__init__(init_cfg)

        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first
        self.temp_encoder = PositionalEncoding(embed_dims)

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)
        # # [num_scale] [bs T cam c h w] # bs dict(0~T-1)
        prev_img_feat = kwargs['prev_img_feat']
        prev_img_metas = kwargs['prev_img_metas']
        len_queue = prev_img_feat[0].shape[1]
        output_history = []

        bs, num_query, _ = query.size()
        query_old = self.temp_encoder(query,1)    # seq_len bs emb_dims
        query_now = self.temp_encoder(query,0)
        attention_weights_old = self.attention_weights(query_old).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        for i in range(len_queue):#actually len_queue=1 right now
            img_metas = [each[i] for each in prev_img_metas]
            img_feats = [each_scale[:, i] for each_scale in prev_img_feat]
            _, output_old, mask_old = feature_sampling(
                img_feats, reference_points, self.pc_range, img_metas) #remember to transform lidar2img@t-1

        attention_weights = self.attention_weights(query_now).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        reference_points_3d, output_now, mask_now = feature_sampling(
                value, reference_points, self.pc_range, kwargs['img_metas'])

        output = torch.cat((output_old, output_now), -2)
        mask = torch.cat((mask_old, mask_now), -2)
        attention_weights = torch.cat((attention_weights_old, attention_weights), -2)
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)
        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)#还调整么。。。后面有ffn按理说应该够用了吧？
        
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
        return self.dropout(output) + inp_residual + pos_feat