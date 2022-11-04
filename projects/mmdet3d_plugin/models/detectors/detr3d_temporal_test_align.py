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
 
    def obtain_history_feat(self, imgs_queue, img_metas_list, test_mode=False):
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, img_metas=None, len_queue=len_queue)
        if test_mode==False:
            self.train()
        # with this open in test time, the self-attn module will fucked up, dropout remains in test time
        # thus leading to different result against detr3d_temporal
        # now we rip it off, and test if results remain the same, actually it goes up, 41.26->47->50
        # now we got object bins of 41&&50, very different but breakpoints got the same shit, maybe its about post-processing
        # >>> cnt
        # 2338800
        # >>> len(objs1.objects)
        # 2399400
        # guess that only every first frame in the scene resembles: (2399400-2338800)/300=202
        return img_feats_list
        
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
        """
        Test function of point cloud branch.
        x: [num_scale] * B cam CHW
        img_metas: [B]
        only support B=1 now
        """
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
    
    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton.""" #only support batchsize=1
        # align with forward_train
        img = img.unsqueeze(0)
        img_metas = [img_metas]

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        prev_img_feat = self.obtain_history_feat(prev_img, prev_img_metas, test_mode = True)#need a different function against forward train
        # prev_bev = None
        img_metas = [each[len_queue-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale, 
            prev_img_feat, prev_img_metas)
        
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list    #list of dict of pts_bbox=dict(bboxes scores labels), len()=batch size