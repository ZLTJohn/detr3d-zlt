@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, use_history=False, pc_range=None, **kwargs):
        super(Detr3DTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        print(kwargs)
        self.use_history = use_history   # test time aug
        self.pc_range = pc_range
        self.prev={'query_output': None, 'refpt': None, 'img_metas': None}

    def check_prev_scene(self, img_metas):
        # only support bs=1 now
        if self.prev['img_metas'] == None: 
            return
        # waymo only
        scene_id = img_metas[0]['sample_idx']//1000
        prev_scene_id = self.prev['img_metas'][0]['sample_idx']//1000
        if scene_id != prev_scene_id:
            self.prev={'query_output': None, 'refpt': None, 'img_metas': None}
        else:
            refpt_prev = self.prev['refpt']   #bs, numq, 3
            prev_img_metas = self.prev['img_metas']   #[bs]
            # img_metas = img_metas
            ego_prev2glob = np.asarray([each['pose'] for each in prev_img_metas]) # bs 4,4
            glob2ego = np.linalg.inv(np.asarray([each['pose'] for each in img_metas]))
            ego_prev2ego = glob2ego @ ego_prev2glob
            ego_prev2ego = refpt_prev.new_tensor(ego_prev2ego)
            pc_range = self.pc_range
            refpt_prev[..., 0:1] = refpt_prev[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
            refpt_prev[..., 1:2] = refpt_prev[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
            refpt_prev[..., 2:3] = refpt_prev[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
            refpt_prev = torch.cat((refpt_prev, torch.ones_like(refpt_prev[..., :1])), -1)# bs numq 4
            B, num_query = refpt_prev.size()[:2]
            ego_prev2ego = ego_prev2ego.view(B, 1, 4, 4).repeat(1, num_query, 1, 1)   # B num_q 4 4
            refpt_prev = torch.matmul(ego_prev2ego, refpt_prev.unsqueeze(-1)).squeeze(-1)
            refpt_prev[..., 0:1] = (refpt_prev[..., 0:1] - pc_range[0])/(pc_range[3] - pc_range[0])
            refpt_prev[..., 1:2] = (refpt_prev[..., 1:2] - pc_range[1])/(pc_range[4] - pc_range[1])
            refpt_prev[..., 2:3] = (refpt_prev[..., 2:3] - pc_range[2])/(pc_range[5] - pc_range[2])
            self.prev['refpt'] = refpt_prev[...,:3]

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
        self.check_prev_scene(kwargs['img_metas'])
        if (self.prev['query_output'] != None) and self.use_history:
            output = self.prev['query_output']
            reference_points = self.prev['refpt']
        else:
            output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
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
                ### reg branch能不断refine ref point，ref point即为box center
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()    #ref point之间不参与back prop，是不是每层有自己的loss？
            # pts = load_pts(kwargs['img_metas'])
            # save_bev(pts, new_reference_points,'debug_refpoint_bev', kwargs['img_metas'][0]['pts_filename'].split('/')[-1]+'_layer{}'.format(lid))
            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.use_history:
            self.prev['query_output']= output
            self.prev['refpt']= reference_points
            self.prev['img_metas'] = kwargs['img_metas']
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points
