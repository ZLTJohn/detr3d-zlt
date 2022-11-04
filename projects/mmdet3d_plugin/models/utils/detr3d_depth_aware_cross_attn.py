
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
from projects.mmdet3d_plugin.models.utils.detr3d_transformer import feature_sampling, inverse_sigmoid, Detr3DTransformer, Detr3DCrossAtten
# def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
# def inverse_sigmoid(x, eps=1e-5):

# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, scale_factor=2):
#         super().__init__()

#         self.conv = nn.Sequential(
#             nn.Linear(in_channels, out_channels),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(out_channels, out_channels),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         return self.conv(x1)

class CamEncode(nn.Module):
    def __init__(self, D, C):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        self.up1 = nn.Linear(self.C, 512)
        self.depthnet = nn.Linear(512, self.D + self.C)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def forward(self, x):
        # Depth
        x = self.up1(x)
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        new_x = x[:, self.D:(self.D + self.C)]
        return depth, new_x

@TRANSFORMER.register_module()
class Detr3DTransformer_depth(Detr3DTransformer):
    def __init__(self,
                **kwargs):
        depth_interval = kwargs.get('depth_interval')
        if depth_interval == None: depth_interval = 1
        self.depth_interval = depth_interval
        super().__init__(**kwargs)

    def init_layers(self):
        super().init_layers()
        self.depth_mlp = CamEncode(self.depth_interval, self.embed_dims)# watch the layer weights when forward, prevent problems

    def forward(self, 
                mlvl_feats,
                query_embed,
                reg_branches=None,
                **kwargs):
        return super().forward(
                mlvl_feats,
                query_embed,
                reg_branches=None,
                depth_branch = self.depth_mlp,
                **kwargs)
    
@ATTENTION.register_module()
class Detr3DCrossAtten_depth(Detr3DCrossAtten):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

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
                depth_branch=None,
                **kwargs):

        # _0 = time.time()
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
        reference_points_3d, output, mask, refpt_cam_depth = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'], return_depth = True)
        # view(B, C, num_query, num_cam,  1, len(mlvl_feats))

        B, C, numq, numc, numpt, lvl = output.shape
        output = output.permute(0,2,3,4,5, 1).reshape(-1,C)         # [B*numq*numc*1*lvl, C] check former part
        depth, out_new = depth_branch(output)                       # [BQCamL, D+C]
        #refpt_cam_depth #[B,numc,numq,1]
        refpt_cam_depth = (refpt_cam_depth-2) / (max(self.pc_range)-2)      # rescale [4m,75m] to [0,1]
        refpt_cam_depth = (refpt_cam_depth - 0.5) * 2               # [0,1] to [-1,1]
        refpt_cam_depth = refpt_cam_depth.permute(0,2,1,3)          # [B,numq,numc,1]
        refpt_cam_depth = refpt_cam_depth.reshape(B*numq*numc, 1).repeat(1,lvl).reshape(-1,1)   
                                                                    # [B*numq*numc*1*lvl, 1]
        refpt_cam_depth = torch.cat((refpt_cam_depth, torch.zeros_like(refpt_cam_depth[..., :1])), -1)
                                                                    # [BQCamL, 2] (1,depth)
        depth = depth[:,None,None,:]                                # [BQCamL,1,1,D] for [B,C,H,W]
        refpt_cam_depth = refpt_cam_depth[:,None,None,:]            # [BQCamL,1,1,2] for [B,1,1,2] #each point to its own depth prob
        refpt_depth_prob = F.grid_sample(depth, refpt_cam_depth,align_corners = True).squeeze()   # [BQCamL]
        output = refpt_depth_prob[:,None] * out_new                          # [BQc1L,C]
        output = output.view(B,numq,numc,numpt,lvl,C).permute(0,5,1,2,3,4)
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)
        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        
        output = self.output_proj(output)#还调整么。。。后面有ffn按理说应该够用了吧？
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
        # print('  '*6+'rest in attn:',time.time()-_0,'ms')
        return self.dropout(output) + inp_residual + pos_feat