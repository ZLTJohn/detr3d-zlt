import time
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


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER.register_module()
class Detr3DTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 selected_feature_level=0,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 decoder=None,
                 **kwargs):
        super(Detr3DTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.selected_feature_level = selected_feature_level
        print("self.selected_feature_level:",self.selected_feature_level)
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m, Detr3DCrossAtten):
                m.init_weight()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def forward(self,
                mlvl_feats,
                query_embed,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                xxxxxx[bs, embed_dims, h, w].xxxxxxxxx
                (B, N, C, H, W).
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        # _=time.time()
        assert query_embed is not None
        if self.num_feature_levels==1:
            # breakpoint()
            mlvl_feats = [mlvl_feats[self.selected_feature_level]]
        bs = mlvl_feats[0].size(0)      #(B, N, C, H, W).
        query_pos, query = torch.split(query_embed, self.embed_dims , dim=1)#所以positional encoding和query embedding是一样长的 ##[num_query, 2c]
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)#  复制扩充成[batch_size,num_query,c]
        query = query.unsqueeze(0).expand(bs, -1, -1)           #同理
        reference_points = self.reference_points(query_pos)     #从positional encoding 里推出ref point
        reference_points = reference_points.sigmoid()           #归一化
        init_reference_out = reference_points
        # __ = time.time()
        # print('  '*4+'preparation in transformer ',__-_,'ms')
        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            **kwargs)

        # print('  '*4+'decoder:',time.time()-__,'ms')
        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(Detr3DTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape self.reference_points = nn.Linear(self.embed_dims, 3)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            # _ = time.time()
            reference_points_input = reference_points
            output = layer(### fucked up in self-attn module
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)
            # __ = time.time()
            # print('  '*5+'decoder layer {}:'.format(lid),__-_,'ms')
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                
                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                ### reg branch能不断refine ref point，ref point即为box center
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()    #ref point之间不参与back prop，是不是每层有自己的loss？
            # pts = load_pts(kwargs['img_metas'])
            # save_bev(pts, new_reference_points,'debug_refpoint_bev', kwargs['img_metas'][0]['pts_filename'].split('/')[-1]+'_layer{}'.format(lid))
            output = output.permute(1, 0, 2)
            # print('  '*5+'layer post process:',time.time()-__,'ms')
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class Detr3DCrossAtten(BaseModule):
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
                 batch_first=False,
                 waymo_with_nuscene=False):
        super(Detr3DCrossAtten, self).__init__(init_cfg)
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
        self.waymo_with_nuscene = waymo_with_nuscene
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
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
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
        # _1 = time.time()
        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        # __ = time.time()
        # print('  '*6+'feature_sampling ',__-_1,'ms')
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)
        if self.waymo_with_nuscene == True:
            num_view = mask.shape[3]
            attention_weights = attention_weights[:,:,:, :num_view,...]
        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        output = self.output_proj(output)#还调整么。。。后面有ffn按理说应该够用了吧？
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
        # print('  '*6+'rest in attn:',time.time()-_0,'ms')
        return self.dropout(output) + inp_residual + pos_feat


def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas, return_depth=False):
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
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    # import cv2,mmcv
    # imgs = [mmcv.imread(name) for name in img_metas[0]['filename']]
    # for i in range(len(imgs)):
    #     if imgs[i].shape != imgs[0].shape:
    #         padded = np.zeros(imgs[0].shape)
    #         padded[:imgs[i].shape[0], :imgs[i].shape[1], :] = imgs[i]
    #         imgs[i] = padded
    # exit_cond=1
    # f = open('debug_image/query_info.txt','w')
    # for i in range(num_query):
    #     cnt = torch.sum(maskvis[0, : ,i] == 1)
    #     if cnt < 0: continue
    #     f.write(str(cnt)+'\n')
    #     f.write(str(maskvis[0, : ,i])+'\n')
    #     exit_cond=1
    #     color = np.random.randint(256, size=3)
    #     color = [int(x) for x in color]
    #     for j in range(num_cam):
    #         pt = (ref_point_visualize[0,j,i]).cpu().detach().numpy()
    #         f.write('query {} - cam{}: {} norm: {}\n'.format(i,j,pt,reference_points_cam[0,j,i])) #(B num_c num_q 2)
    #         f.write(str(maskvis.shape)+'\n')
    #         f.write(str(ref_point_visualize.shape)+'\n')
    #         if (maskvis[0,j,i] == 1):
    #             cv2.circle(imgs[j], (int(pt[0]),int(pt[1])), radius=5 , color = color, thickness = 4)
    #             cv2.putText(imgs[j], str(i), (int(pt[0]),int(pt[1])),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=color)
    # if exit_cond:
    #     for i in range(num_cam): 
    #         mmcv.imwrite(imgs[i], 'debug_image/nuscene_layer1_refpoint_vis_{}.png'.format(i))
    #     exit(0)
    # print(img_metas)
    if return_depth == True:
        return reference_points_3d, sampled_feats, mask, refpt_depth
    else:
        return reference_points_3d, sampled_feats, mask

def load_pts(img_metas):
    path = img_metas[0]['pts_filename']
    points = np.fromfile(path, dtype=np.float32)
    dim = 6
    if path.find('waymo') == -1:
        dim=5
    return points.reshape(-1, dim)
    
def save_bev(pts , ref, data_root, out_name = None):
    import time
    import torchvision.utils as vutils
    if isinstance(pts, list):
        pts = pts[0]
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    pc_range=[-75, -75, -2, 75, 75, 4]
    mask = ((pts[:, 0] > pc_range[0]) & (pts[:, 0] < pc_range[3]) & 
        (pts[:, 1] > pc_range[1]) & (pts[:, 1] < pc_range[4]) &
        (pts[:, 2] > pc_range[2]) & (pts[:, 2] < pc_range[5]))
    pts = pts[mask]
    res = 0.1
    x_max = 1 + int((pc_range[3] - pc_range[0]) / res)
    y_max = 1 + int((pc_range[4] - pc_range[1]) / res)
    im = torch.zeros(x_max+1, y_max+1, 3)
    x_img = (pts[:, 0] - pc_range[0]) / res
    x_img = x_img.round().long()
    y_img = (pts[:, 1] - pc_range[1]) / res
    y_img = y_img.round().long()
    im[x_img, y_img, :] = 1

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            im[(x_img.long()+i).clamp(min=0, max=x_max), 
                (y_img.long()+j).clamp(min=0, max=y_max), :] = 1
    print('reference', ref.size())
    ref_pts_x = (ref[..., 0] * (pc_range[3] - pc_range[0]) / res).round().long()
    ref_pts_y = (ref[..., 1] * (pc_range[4] - pc_range[1]) / res).round().long()
    for i in [-2, 0, 2]:
        for j in [-2, 0, 2]:
            im[(ref_pts_x.long()+i).clamp(min=0, max=x_max), 
                (ref_pts_y.long()+j).clamp(min=0, max=y_max), 0] = 1
            im[(ref_pts_x.long()+i).clamp(min=0, max=x_max), 
                (ref_pts_y.long()+j).clamp(min=0, max=y_max), 1:2] = 0
    im = im.permute(2, 0, 1)
    timestamp = str(time.time())
    print(timestamp)
    # saved_root = '/home/chenxy/mmdetection3d/'
    if out_name == None:
        out_name = data_root + '/' + timestamp + '.jpg'
    else :
        out_name = data_root + '/' + out_name + '.jpg'
    vutils.save_image(im, out_name)