import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiheadAttention,
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

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DTransformerDecoder_T2(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(Detr3DTransformerDecoder_T2, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.prev={'query_output': None, 'refpt': None, 'img_metas': None}
        # self.temporal_pos_encoder = nn.Linear(3, self.embed_dims)
        self.temporal_pos_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

    def check_prev_scene(self, img_metas):
        # only support bs=1 now
        if self.prev['img_metas'] == None: 
            return
        scene_id = img_metas[0]['sample_idx']//1000
        prev_scene_id = self.prev['img_metas'][0]['sample_idx']//1000
        if scene_id != prev_scene_id:
            breakpoint()
            self.prev={'query_output': None, 'refpt': None, 'img_metas': None}

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
        self.check_prev_scene(kwargs['img_metas'])
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                prev_query_output = self.prev['query_output'],
                prev_refpt = self.prev['refpt'],
                prev_img_metas = self.prev['img_metas'],
                temporal_pos_encoder = self.temporal_pos_encoder,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                
                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()    #ref point之间不参与back prop，是不是每层有自己的loss？
            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        # breakpoint()
        self.prev['query_output']= output
        self.prev['refpt']= reference_points
        self.prev['img_metas'] = kwargs['img_metas']
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points

@ATTENTION.register_module()
class Detr3DTemporalCrossAttn(MultiheadAttention):
    def __init__(self,
                 **kwargs):
        super(Detr3DTemporalCrossAttn, self).__init__(**kwargs)

    def forward(self,
                query,
                key=None,       # key is last frame query, you should input it in transformer level
                value=None,
                identity=None,
                query_pos=None, #generated by ref points here
                key_pos=None,   #key pos also in outer layer
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        #kwargs['img_metas']
        #value = key = last frame query
        # breakpoint()
        # print('------------------',kwargs,'--------------')
        if kwargs['prev_refpt'] == None:
            return query
        
        refpt_prev = kwargs['prev_refpt']   #bs, numq, 3
        prev_img_metas = kwargs['prev_img_metas']   #[bs]
        img_metas = kwargs['img_metas']
        ego_prev2glob = np.asarray([each['pose'] for each in prev_img_metas]) # bs 4,4
        glob2ego = np.linalg.inv(np.asarray([each['pose'] for each in img_metas]))
        ego_prev2ego = glob2ego @ ego_prev2glob
        ego_prev2ego = refpt_prev.new_tensor(ego_prev2ego)

        refpt_prev = torch.cat((refpt_prev, torch.ones_like(refpt_prev[..., :1])), -1)# bs numq 4
        B, num_query = refpt_prev.size()[:2]
        # breakpoint()
        ego_prev2ego = ego_prev2ego.view(B, 1, 4, 4).repeat(1, num_query, 1, 1)   # B num_q 4 4
        refpt_prev = torch.matmul(ego_prev2ego, refpt_prev.unsqueeze(-1)).squeeze(-1)
        refpt = kwargs['reference_points']
        temporal_pos_encoder = kwargs['temporal_pos_encoder']#input it in detr3d_temporal2.py

        value = kwargs['prev_query_output']
        key = value
        key_pos = temporal_pos_encoder(refpt_prev[...,:3]).permute(1,0,2)
        query_pos = temporal_pos_encoder(refpt).permute(1,0,2)
        
        # return super(Detr3DTemporalCrossAttn, self).forward(
        #             query,
        #             key,
        #             value,
        #             identity if self.pre_norm else None,
        #             query_pos=query_pos,
        #             key_pos=key_pos,
        #             attn_mask=attn_masks[attn_index],
        #             key_padding_mask=key_padding_mask,
        #             **kwargs)

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.dropout_layer(self.proj_drop(out))
        
       
#  query = self.attentions[attn_index](
#                     query,
#                     key,
#                     value,
#                     identity if self.pre_norm else None,
#                     query_pos=query_pos,
#                     key_pos=key_pos,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=key_padding_mask,
#                     **kwargs)