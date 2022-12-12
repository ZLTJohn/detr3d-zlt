
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
from projects.mmdet3d_plugin.models.utils.detr3d_transformer import feature_sampling, inverse_sigmoid,Detr3DCrossAtten
# def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
# def inverse_sigmoid(x, eps=1e-5):
@ATTENTION.register_module()
class Detr3DCrossAtten_T3(Detr3DCrossAtten):
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
        # import ipdb
        # ipdb.set_trace()
        num_q = query.shape[0]
        query_prev = super(Detr3DCrossAtten_T3,self).forward(
                query[:num_q//2,...],
                key,
                [each_scale[:, 0] for each_scale in kwargs['prev_img_feat']],
                residual,
                query_pos[:num_q//2,...],
                key_padding_mask,
                reference_points[:,:num_q//2,:],
                spatial_shapes,
                level_start_index,
                img_metas = [kwargs['prev_img_metas'][0][0]])

        query_curr = super(Detr3DCrossAtten_T3,self).forward(
                query[num_q//2:,...],
                key,
                value,
                residual,
                query_pos[num_q//2:,...],
                key_padding_mask,
                reference_points[:,num_q//2:,:],
                spatial_shapes,
                level_start_index,
                img_metas = kwargs['img_metas'])
        return torch.cat((query_prev,query_curr),0)