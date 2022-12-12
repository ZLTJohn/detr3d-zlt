import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscnes_eval import NuScenesEval_custom
# from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random
import nuscenes.utils.geometry_utils as nug_utils
from .zltvis_temporal import save_temporal_frame
from .nus_dataset_T import CustomNuScenesDataset_T
@DATASETS.register_module()
class CustomNuScenesDataset_T_test_align(CustomNuScenesDataset_T):
    r"""NuScenes Dataset.
    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.history_len = self.queue_length-1
        
    def prepare_test_data(self, index):
        idx_list = list(range(index-self.history_len, index))  + [index]
        idx_list = sorted(idx_list, reverse=True)
        data_queue = []
        for i in idx_list:
            i = max(0,i)
            input_dict = self.get_data_info(i)
            if (input_dict != None):
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                example = {'img':example['img'][0], 'img_metas':example['img_metas'][0]}
            data_queue.insert(0, copy.deepcopy(example))
        #data_queue: T-len+1, T

        return self.union2one(data_queue)