import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
@PIPELINES.register_module()
class ProjectLabelToWaymoClass(object):
    def __init__(self, class_names = None, waymo_name = [ 'Car', 'Pedestrian', 'Cyclist']):
        self.class_names = class_names
        self.waymo_name = waymo_name
        self.name_map = {
            'car'                   :   'Car',
            'truck'                 :   'Car',
            'construction_vehicle'  :   'Car',
            'bus'                   :   'Car',
            'trailer'               :   'Car',
            'motorcycle'            :   'Car',
            'bicycle'               :   'Cyclist',
            'pedestrian'            :   'Pedestrian'
        }
        ind_N2W=[]
        for i,n_name in enumerate(self.class_names): ind_N2W.append(waymo_name.index(self.name_map[n_name]))
        self.ind_N2W = np.array(ind_N2W)
    # class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    #                'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    def __call__(self, results):
        if len(results['gt_labels_3d']) > 0:
            results['gt_labels_3d'] = self.ind_N2W[results['gt_labels_3d']]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str