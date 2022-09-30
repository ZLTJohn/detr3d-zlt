import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from projects.mmdet3d_plugin.models.utils.detr3d_transformer import feature_sampling, inverse_sigmoid, Detr3DCrossAtten

@ATTENTION.register_module()
class Detr3DCrossAtten_deform3d(Detr3DCrossAtten):

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=7,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 offset_size = 0.05,#leftermost to rightermost of grid
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False,
                 use_dynamic = False):
        super(Detr3DCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
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
        self.use_dynamic = use_dynamic
        if use_dynamic:
            self.sampling_offsets3d = nn.Linear(embed_dims, num_points*3)
        self.offset_size = offset_size
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

        self.init_weight()

    def init_weight(self):
        # x=[1,4,7]  y=[0.8,1.8,2.8] z=[1,2,3]   mean size[3.51, 1.77, 1.69]
        """Default initialization for Parameters of Module."""
        x,y,z = 3.51*0.5/(75+35), 1.77*0.5/(75+75), 1.69*0.5/(4+2)
        self.grid_init = torch.tensor([[x,0,0],[-x,0,0],
                                       [0,y,0],[0,-y,0],
                                       [0,0,z],[0,0,-z],[0,0,0]])#[num_pt,3]
        if self.use_dynamic:
            constant_init(self.sampling_offsets3d, 0.)
            # self.sampling_offsets3d.bias.data = self.grid_init.view(-1)
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

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        ref_pt_offset = None
        if self.use_dynamic:
            ref_pt_offset = self.sampling_offsets3d(query).view(bs, num_query, self.num_points, 3)  #B,numq,3
        else:
            ref_pt_offset =self.grid_init.clone().cuda().view(1,1,7,3).repeat(bs,num_query,1,1)

        reference_points_3d = reference_points.clone()
        output, mask= [], []
        for i in range(self.num_points):
            shifted_refpt = reference_points + ref_pt_offset[...,i,:]
            refpt_3d, output_i, mask_i = feature_sampling(
                value, shifted_refpt, self.pc_range, kwargs['img_metas'])
            output.append(output_i)
            mask.append(mask_i)
        output = torch.nan_to_num(torch.cat(output,-2))
        mask = torch.nan_to_num(torch.cat(mask,-2))
        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        output = self.output_proj(output)#还调整么。。。后面有ffn按理说应该够用了吧？
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat
