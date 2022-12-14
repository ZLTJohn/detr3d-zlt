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
"""
detr3d_temporal now only support single gpu evaluation since the distributed data sampler
has problem that samples are not distributed in sequence in one gpu, leading to uncontrollable
temporal interval
"""
@DETECTORS.register_module()
class Detr3D_T(MVXTwoStageDetector):
    """Detr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 debug_name=None,
                 debug_dir='debug/2Hz_recheck/',
                 gtvis_range = [0,105],
                 vis_count=30):
        super(Detr3D_T,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev = {'img_feats': None, 'img_metas':None, 'scene_id':None}
        self.debug_name = debug_name
        self.debug_dir = debug_dir
        if torch.distributed.get_rank()==0 and debug_dir!=None:
            if not os.path.exists(debug_dir):
                os.mkdir(debug_dir)
        self.gtvis_range = gtvis_range
        self.vis_count = vis_count

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            # input_shape = img.shape[-2:]#bs nchw
            # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)       # mask out some grids
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas, len_queue=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas, len_queue)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_img_feat=None, 
                          prev_img_metas=None
                          ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_img_feat = prev_img_feat, prev_img_metas = prev_img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

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

    def forward_train(self,
                      points=None,
                      img_metas=None,##
                      gt_bboxes_3d=None,##
                      gt_labels_3d=None,##
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,##
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        prev_img_feat = self.obtain_history_feat(prev_img, prev_img_metas)
        # prev_bev = None
        img_metas = [each[len_queue-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        if self.debug_name!= None:
            dir = self.debug_dir
            name = str(img_metas[0]['sample_idx'])
            if os.path.exists(dir + name) == False:
                os.mkdir(dir + name)
            visualize_gt(gt_bboxes_3d, gt_labels_3d, img_metas,
                    self.debug_name,
                    dir_name = dir + name,
                    gtvis_range = self.gtvis_range)
            visualize_gt(gt_bboxes_3d, gt_labels_3d, [prev_img_metas[0][0]],
                    self.debug_name+'_prev',
                    dir_name = dir + name,
                    gtvis_range = self.gtvis_range)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_img_feat, prev_img_metas)
                                            # [num_scale] [bs T cam c h w] # bs dict(0~T-1)

        losses.update(losses_pts)
        return losses
    # test is yet to be changed
    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def update_prev(self, x, img_metas, Force = True):
        scene_id = [ (each['sample_idx']//1000) for each in img_metas]
        bs = len(img_metas)
        if Force == True:
            self.prev['scene_id'] = copy.deepcopy(scene_id)
            self.prev['img_feats'] = [each.unsqueeze(1) for each in x]
            self.prev['img_metas'] = [{0: copy.deepcopy(each)} for each in img_metas]
        else:
            for i in range(bs):
                if self.prev['scene_id'][i]!=scene_id[i]:
                    self.prev['scene_id'][i] = copy.deepcopy(scene_id[i])
                    for lvl in range(len(x)):   self.prev['img_feats'][lvl][i] = x[lvl][i].unsqueeze(0)
                    self.prev['img_metas'][i] = {0: copy.deepcopy(img_metas[i])}
                else:#update lidar2img
                    len_queue = 1
                    glob2ego_prev = np.linalg.inv(self.prev['img_metas'][i][len_queue-1]['pose'])
                    ego2glob = img_metas[i]['pose']
                    # ego2img_old =ego_old2img_old @ global2ego_old @ ego2global #@pt_ego
                    for key in range(len_queue):
                        ego_prev2img_i = self.prev['img_metas'][i][key]['lidar2img']
                        ego2img_i = ego_prev2img_i @ glob2ego_prev @ ego2glob#@pt_ego
                        self.prev['img_metas'][i][key]['lidar2img'] = copy.deepcopy(ego2img_i)

    def simple_test_pts(self, x, img_metas, rescale=False):
        """
        Test function of point cloud branch.
        x: [num_scale] * B cam CHW
        img_metas: [B]
        only support B=1 now
        """
        # outs = self.pts_bbox_head(pts_feats, img_metas, prev_img_feat, prev_img_metas)
        # breakpoint() #nothing wrong compared to test_align
        if self.prev['scene_id'] != None:
            print('\nbefoer update:gpu: {}, img_feat devices: {}, prev_feat devices: {}, prev_frame: {}, curr_frame: {}'.format(torch.distributed.get_rank(),x[0].device,self.prev['img_feats'][0].device,self.prev['img_metas'][0][0]['sample_idx'],img_metas[0]['sample_idx']))
        self.update_prev(x, img_metas, Force = (self.prev['scene_id'] == None))
        print('\nafter update gpu: {}, img_feat devices: {}, prev_feat devices: {}, prev_frame: {}, curr_frame: {}'.format(torch.distributed.get_rank(),x[0].device,self.prev['img_feats'][0].device,self.prev['img_metas'][0][0]['sample_idx'],img_metas[0]['sample_idx']))

        outs = self.pts_bbox_head(x, img_metas, prev_img_feat = self.prev['img_feats'], prev_img_metas = self.prev['img_metas'])
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        # breakpoint() # result differs
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)       # to CPU
            for bboxes, scores, labels in bbox_list     #for each in batch
        ]
        return bbox_results #list of dict(bboxes scores labels) in one frame
    
    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)

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
                
            save_bbox_pred(bbox_pts, img, [self.prev['img_metas'][0][0]],
                self.debug_name+'_prev', 
                dir_name = dir + name, 
                vis_count = self.vis_count)
        
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        self.update_prev(img_feats, img_metas, Force = True)
        return bbox_list    #list of dict of pts_bbox=dict(bboxes scores labels), len()=batch size