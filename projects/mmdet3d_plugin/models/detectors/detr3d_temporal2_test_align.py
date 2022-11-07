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
from .detr3d_temporal2 import Detr3D_T2
@DETECTORS.register_module()
class Detr3D_T2_test_align(Detr3D_T2):
    """Detr3D."""

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
 
    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test_pts(self, x, img_metas_T, rescale=False):
        """Test function of point cloud branch."""
        len_queue = x[0].shape[1]
        for i in range(len_queue):
            feats_i = [each_scale[:,i] for each_scale in x]
            img_metas = [each[i] for each in img_metas_T]
            # breakpoint()
            outs = self.pts_bbox_head(feats_i, img_metas, clear_prev = (i==0) )

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)       # to CPU
            for bboxes, scores, labels in bbox_list     #for each in batch
        ]
        return bbox_results #list of dict(bboxes scores labels) in one frame

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img = img.unsqueeze(0)
        img_metas = [img_metas]
        
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list    #list of dict of pts_bbox=dict(bboxes scores labels), len()=batch size