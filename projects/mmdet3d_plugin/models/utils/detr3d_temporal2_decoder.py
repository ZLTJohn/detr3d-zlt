@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DTransformerDecoder_T2(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(Detr3DTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.prev={'query_output': None, 'refpt': None, 'img_metas': None}
        self.temporal_pos_encoder = nn.Linear(3, self.embed_dims)
    
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
        if kwargs['prev_refpt'] == None: return query
        
        refpt_prev = kwargs['prev_refpt']   #bs, numq, 3
        prev_img_metas = kwargs['prev_img_metas']   #[bs]
        ego_prev2glob = np.asarray([each['pose'] for each in prev_img_metas]) # bs num_cam 4,4
        glob2ego = np.linalg.inv(np.asarray(kwargs['img_metas']['pose']))
        ego_prev2ego = glob2ego @ ego_prev2glob
        glob2ego = refpt_prev.new_tensor(glob2ego)

        refpt_prev = torch.cat((refpt_prev, torch.ones_like(refpt_prev[..., :1])), -1)
        refpt_prev = torch.matmul(ego_prev2ego, refpt_prev)
        refpt = kwargs['reference_points']
        temporal_pos_encoder = kwargs['temporal_pos_encoder']#input it in detr3d_temporal2.py

        value = kwargs['prev_query_output']
        key = value
        key_pos = temporal_pos_encoder(refpt_prev[...,:3])
        query_pos = temporal_pos_encoder(refpt)
        return super(Detr3DTemporalCrossAttn, self).forward(
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
       

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