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
class Detr3D_T_back_history(Detr3D_T_test_align):
    """Detr3D."""
    def obtain_history_feat(self, imgs_queue, img_metas_list, test_mode=False):
        prev_bev = None
        bs, len_queue, num_cams, C, H, W = imgs_queue.shape
        imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
        img_feats_list = self.extract_feat(img=imgs_queue, img_metas=None, len_queue=len_queue)
        return img_feats_list