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
import einops
from projects.mmdet3d_plugin.models.utils.detr3d_transformer import inverse_sigmoid

@ATTENTION.register_module()
class Detr3DCrossAtten_Kernel(BaseModule):

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                #  num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 kernel_height=1,
                 kernel_width=1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DCrossAtten_Kernel, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        print('DETR3D CrossAttn pc_range: {}'.format(pc_range))
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
        # self.num_points = num_points
        self.kh = kernel_height
        self.kw = kernel_width
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*\
                                           kernel_height*kernel_width)

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

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.kh*self.kw , self.num_levels)
        
        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'], self.kh, self.kw)
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        
        output = self.output_proj(output)#还调整么。。。后面有ffn按理说应该够用了吧？
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat


def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas, k_h ,k_w):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    # import hashlib
    # print(lidar2img[0,0])
    # print(hashlib.md5(lidar2img[0,0]).hexdigest())
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    if pc_range[-1] == 'polar_coordinates':
        thetas = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
        radii =  reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 0:1] = torch.cos(thetas) * radii
        reference_points[..., 1:2] = torch.sin(thetas) * radii
        reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    else:
        reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4) ，to homogeneous coordinate
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)##B num_c num_q 4 ,1
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)   # B num_c num_q 4 4
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)    # B num_c num_q 4
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)   #filter out negative depth, B num_c num_q
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(  #z for depth, too shallow will cause zero division
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)    # eps controls minimum

    # ref_point_visualize = reference_points_cam.clone()
    #try to normalize to the coordinate in feature map
    if type(img_metas[0]['ori_shape']) == tuple:    
        #same size for all images, nuscene  900*1600,floor to 928*1600
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    else:   
        #diff size,1280*1920 and 886*1920, waymo, get it to the normorlized point, floor 886 to 896 to meet divisor 32, 
        # which is 0.7 out of 1 against 1280, that is to say, the remaining 30% is padding
        reference_points_cam[..., 0] /= img_metas[0]['ori_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['ori_shape'][0][0]
        mask[:, 3:5, :] &= (reference_points_cam[:, 3:5, :, 1:2] < 0.7)

    reference_points_cam = (reference_points_cam - 0.5) * 2     #0~1 to -1~1
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)  #we should change the criteria for waymo cam 3~4
                 & (reference_points_cam[..., 0:1] < 1.0)   # which is -1~0.4
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    # maskvis = mask.clone()
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)

 
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        ref_pt =  reference_points_cam.view(B*N, num_query, 1, 2)
        ref_pt =  ref_pt.repeat(1, 1, k_h*k_w ,1)         #[bn q kh*kw 2]

        kernel_offset = generate_grid(k_h, k_w, 2.0/H, 2.0/W)  # scale: 0~H,0~W -> -1~1, -1~1
        ref_pt_offset = kernel_offset.view(1, 1, k_h*k_w, 2)
        ref_pt_offset = ref_pt_offset.repeat(B*num_cam, num_query, 1, 1)     #[bn q kh*kw 2]
        ref_pt_kernel = ref_pt + ref_pt_offset
        sampled_feat = F.grid_sample(feat, ref_pt_kernel)

        sampled_feat = sampled_feat.view(B, N, C, num_query, k_h*k_w).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  k_h*k_w, len(mlvl_feats))

    return reference_points_3d, sampled_feats, mask

def generate_grid(height: int, width: int, scaleh = 1, scalew = 1):#scale factor: 单位一
    xs = torch.linspace(-(width//2), width//2, width) * scalew
    ys = torch.linspace(-(height//2), height//2, height) * scaleh
    indices = torch.stack(torch.meshgrid((xs, ys)), -1)#h w 2
    return indices.cuda()