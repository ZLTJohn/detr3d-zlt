import cv2
import numpy as np
import time
import torchvision.utils as vutils
import torch
import copy
import os
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from .visualizer_zlt import *
from .detr3d_temporal_test_align import Detr3D_T_test_align
@DETECTORS.register_module()
class Detr3D_T3(Detr3D_T_test_align):
    """Detr3D."""

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
    def divide_output(self,outs):
        out_prev, out_curr={},{}
        num_q = self.pts_bbox_head.num_query
        #dict_keys(['all_cls_scores', 'all_bbox_preds', 'enc_cls_scores'->none, 'enc_bbox_preds'->none])
        for key in outs: 
            if outs[key]!=None: 
                out_prev[key] = outs[key][:,:,:num_q//2,:]#num_layer,bs,query,
                out_curr[key] = outs[key][:,:,num_q//2:,:]
            else:
                out_prev[key] = None
                out_curr[key] = None
        return out_prev, out_curr
    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_img_feat=None, 
                          prev_img_metas=None
                          ):
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_img_feat = prev_img_feat, prev_img_metas = prev_img_metas)
        out_prev, out_curr = self.divide_output(outs)
        device = pts_feats[0].device
        # print(device,'rank: ',torch.distributed.get_rank())
        loss_inputs_prev = [[gt_bboxes_3d[0][0].to(device)], [gt_labels_3d[0][0].to(device)], out_prev]
        losses_prev = self.pts_bbox_head.loss(*loss_inputs_prev)
        loss_inputs_curr = [[gt_bboxes_3d[0][1].to(device)], [gt_labels_3d[0][1].to(device)], out_curr]
        losses_curr = self.pts_bbox_head.loss(*loss_inputs_curr)
        losses = {}
        for key in losses_prev:
            losses[key] = (losses_prev[key] + losses_curr[key]) / 2
        return losses

    def simple_test_pts(self, x, img_metas, rescale=False,
                        prev_img_feat=None, 
                        prev_img_metas=None):
        # breakpoint()
        outs = self.pts_bbox_head(x, img_metas, prev_img_feat = prev_img_feat,prev_img_metas = prev_img_metas)
        out_prev, out_curr = self.divide_output(outs)
        outs = out_curr
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        # breakpoint()
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)       # to CPU
            for bboxes, scores, labels in bbox_list     #for each in batch
        ]
        return bbox_results #list of dict(bboxes scores labels) in one frame