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
from .detr3d_temporal import Detr3D_T
@DETECTORS.register_module()
class Detr3D_T_test_align(Detr3D_T):
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

    def simple_test_pts(self, x, img_metas, rescale=False,
                        prev_img_feat=None, 
                        prev_img_metas=None):
        # breakpoint()
        outs = self.pts_bbox_head(x, img_metas, prev_img_feat = prev_img_feat,
                                                prev_img_metas = prev_img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        # breakpoint()
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)       # to CPU
            for bboxes, scores, labels in bbox_list     #for each in batch
        ]
        return bbox_results #list of dict(bboxes scores labels) in one frame
    
    def simple_test(self, img_metas, img=None, rescale=False):  #only support batchsize=1
        # align with forward_train
        img = img.unsqueeze(0)
        img_metas = [img_metas]

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        prev_img_feat = self.obtain_history_feat(prev_img, prev_img_metas, test_mode = True)

        img_metas = [each[len_queue-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale, 
            prev_img_feat, prev_img_metas)
        if self.debug_name != None:
            # name = str(time.time())
            # breakpoint()
            dir = self.debug_dir
            name = str(img_metas[0]['sample_idx'])
            if os.path.exists(dir + name) == False:
                os.mkdir(dir + name)
            save_bbox_pred(bbox_pts, img, img_metas,
                self.debug_name,
                dir_name = dir + name, 
                vis_count = self.vis_count)
            print('curr: {}, prev: {}'.format(name,prev_img_metas[0][0]['sample_idx']))
            save_bbox_pred(bbox_pts, img, [prev_img_metas[0][0]],
                self.debug_name+'_prev', 
                dir_name = dir + name, 
                vis_count = self.vis_count)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list    #list of dict of pts_bbox=dict(bboxes scores labels), len()=batch size