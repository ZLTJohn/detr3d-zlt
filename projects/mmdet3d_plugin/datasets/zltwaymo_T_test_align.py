import mmcv
import numpy as np
import os
import tempfile
import torch
from mmcv.utils import print_log
from os import path as osp
# ERROR ROOT at LINE 331, AT line 236 in format_result, we adjust the worker to be really small
from mmdet3d.datasets import DATASETS #really fucked up for not adding '3d'
from mmdet3d.core.bbox import Box3DMode, points_cam2img
from mmdet3d.datasets.kitti_dataset import KittiDataset
from .waymo_let_metric import compute_waymo_let_metric
import copy
from mmcv.parallel import DataContainer as DC
import random

from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from .zltvis_temporal import save_temporal_frame
from .zltwaymo_temporal import CustomWaymoDataset_T
@DATASETS.register_module()
class CustomWaymoDataset_T_test_align(CustomWaymoDataset_T):

    CLASSES = ('Car', 'Pedestrian', 'Sign', 'Cyclist')

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_test_data(self, index):
        idx_list = list(range(index-self.history_len, index))  + [index]
        idx_list = sorted(idx_list, reverse=True)
        data_queue = []
        scene_id = None
        for i in idx_list:
            i = max(0,i)
            input_dict = self.get_data_info(i)
            if scene_id == None: scene_id = input_dict['sample_idx']//1000
            if (input_dict != None) and (scene_id == input_dict['sample_idx']//1000):
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                example = {'img':example['img'][0], 'img_metas':example['img_metas'][0]}
            data_queue.insert(0, copy.deepcopy(example))
        #data_queue: T-len+1, T
        return self.union2one(data_queue)