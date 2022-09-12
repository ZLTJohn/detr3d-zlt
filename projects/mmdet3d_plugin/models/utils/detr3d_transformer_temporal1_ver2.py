
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
from projects.mmdet3d_plugin.models.utils.detr3d_transformer import feature_sampling, inverse_sigmoid
# def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
# def inverse_sigmoid(x, eps=1e-5):

@ATTENTION.register_module()
class Detr3DCrossAtten_T1v2(BaseModule):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

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
        super(Detr3DCrossAtten_T1v2, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        # print('DETR3D CrossAttn pc_range: {}'.format(pc_range))
        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points*2)
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
        # breakpoint()

        bs, num_query, _ = query.size()
        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points*2, self.num_levels)

        for i in range(len_queue):#actually len_queue=1 right now
            img_metas = [each[i] for each in prev_img_metas]
            img_feats = [each_scale[:, i] for each_scale in prev_img_feat]
            _, output_old, mask_old = feature_sampling(
                img_feats, reference_points, self.pc_range, img_metas) #remember to transform lidar2img@t-1

        reference_points_3d, output_now, mask_now = feature_sampling(
                value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.cat((output_old, output_now), -2)
        mask = torch.cat((mask_old, mask_now), -2)
        breakpoint()
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)
        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)#还调整么。。。后面有ffn按理说应该够用了吧？
        
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
        return self.dropout(output) + inp_residual + pos_feat
