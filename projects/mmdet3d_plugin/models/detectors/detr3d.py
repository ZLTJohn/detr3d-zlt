import cv2
import numpy as np
import time
import torchvision.utils as vutils
import torch

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class Detr3D(MVXTwoStageDetector):
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
                 pretrained=None):
        super(Detr3D,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]#bs nchw
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

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
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
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
        # print(img_metas[0])
        # exit(0)
        outs = self.pts_bbox_head(pts_feats, img_metas)
        # bbox_list = self.pts_bbox_head.get_bboxes(
        #     outs, img_metas, rescale=False)
        # import cv2
        # for i,name in enumerate(img_metas[0]['filename']):
        #     img = cv2.imread(name)
        #     cv2.imwrite('debug_target/gt_vis_{}.png'.format(i),img)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        # print(bbox_list)
        # print(bbox_list[0][2].size())#300
        # print(losses)
        # exit(0)
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
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        # open('debug_forward/waymo_train_img_metas.txt','w').write(str(img_metas)+'\ninput img shape:'+str(img.shape))
        # name = str(time.time())
        # save_bbox2img(img, gt_bboxes_3d, img_metas, name = name)
        # save_bbox2bev(gt_bboxes_3d, img_metas, name=name)
        # exit(0)
        # _ = time.time()
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # __ = time.time()
        # print('  '*2+'extract_feat: ',__-_,'ms')
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        # print('  '*2+'forward_pts_train: ',time.time()-__,'ms')
        losses.update(losses_pts)
        return losses
    
    def forward_test(self, img_metas, img=None, **kwargs):
        # open('debug_forward/waymo_test_img_metas.txt','w').write(str(img_metas)+'\ninput img shape:'+str(img[0].shape)+'\nlen of img list'+str(len(img)))
        # exit(0)
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)
        # if num_augs == 1:
        #     img = [img] if img is None else img
        #     return self.simple_test(None, img_metas[0], img[0], **kwargs)
        # else:
        #     return self.aug_test(None, img_metas, img, **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        # print(bbox_list)
        # print(bbox_list[0][2].size())#300
        # exit(0)
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
        # save_bbox_pred(bbox_pts, img, img_metas)
        
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list    #list of dict of pts_bbox=dict(bboxes scores labels), len()=batch size

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


def save_bbox2img(img, gt_bboxes_3d, img_metas, dirname='debug_coord', name = None):
    # print(img)
    ds_name = 'waymo' if len(img_metas[0]['filename'])==5 else 'nuscene'
    # for i,name in enumerate(img_metas[0]['filename']):
    #     img_out = img[0][i].permute(1,2,0).detach().cpu().numpy()
    #     img_in = cv2.imread(name)
    #     print(img_in)
    #     print(img_out.shape)
    #     cv2.imwrite('debug_forward/{}_input_vis_finalcheck_{}.png'.format(ds_name, i),img_out)
    gt_bboxes_3d = gt_bboxes_3d[0]
    reference_points = gt_bboxes_3d.gravity_center.view(1, -1, 3) # 1 num_gt, 3
    # reference_points = gt_bboxes_3d.bottom_center.view(1, -1, 3)
    print(reference_points.size())
    # num_gt as num_query
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    # reference_points (B, num_queries, 4) ，to homogeneous coordinate
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)

    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)##B num_c num_q 4 ,1
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)   # B num_c num_q 4 4
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)    # B num_c num_q 4
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)   #filter out negative depth, B num_c num_q
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(  #z for depth, too shallow will cause zero division
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)    # eps controls minimum

    # ref_point_visualize = reference_points_cam.clone()
    #try to normalize to the coordinate in feature map
    if type(img_metas[0]['ori_shape']) == tuple:    
        #same size for all images, nuscene  900*1600,floor to 928*1600
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    else:
        #diff size,1280*1920 and 886*1920, waymo, get it to the normorlized point, floor 886 to 896 to meet divisor 32, 
        # which is 0.7 out of 1 against 1280, that is to say, the remaining 30% is padding
        reference_points_cam[..., 0] /= img_metas[0]['ori_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['ori_shape'][0][0]
        print(img_metas[0]['ori_shape'])
        mask[:, 3:5, :] &= (reference_points_cam[:, 3:5, :, 1:2] < 0.7)

    # reference_points_cam = (reference_points_cam - 0.5) * 2       #0~1 to -1~1
    mask = (mask & (reference_points_cam[..., 0:1] > 0)  #we should change the criteria for waymo cam 3~4
                 & (reference_points_cam[..., 0:1] < 1)   # which is -1~0.4
                 & (reference_points_cam[..., 1:2] > 0) 
                 & (reference_points_cam[..., 1:2] < 1))  # B num_cam num_query
    print(mask.any()==False)
    # print(lidar2img.shape)# 1 5 4 4
    # print(img)      #shape 1 5 3 640 960s
    # print(gt_bboxes_3d.gravity_center)#[x,y,z,w,l,h,rot] #z is not gravity center
    if (name == None):
        name = str(time.time())
    for i in range(num_cam):
        # img_out = img[0][i].permute(1,2,0).detach().cpu().numpy()
        img_out = cv2.imread(img_metas[0]['filename'][i])
        h,w,_ = img_metas[0]['ori_shape'][0]#img_out.shape
        print(h,w)
        # print(img_out)
        # print(img_in)

        for j in range(num_query):
            pt = reference_points_cam[0,i,j]
            print(pt,'  ',mask[0,i,j])
            if mask[0,i,j] == True:
                color = np.random.randint(256, size=3)
                color = [int(x) for x in color]
                cv2.circle(img_out, (int(pt[0]*w),int(pt[1]*h)), radius=5,color=color, thickness = 4)
        cv2.imwrite(dirname+'/{}_{}_{}.png'.format(ds_name,name, i), img_out)

def save_bbox_pred(bbox_pts, img, img_metas):
    # print(bbox_pts[0])## get our favourite frame id=475 in zltwaymo
    print(len(bbox_pts[0]))
    # print(bbox_pts[0]['boxes_3d'])
    # np.save(dir+'/bbox_pred',bbox_pts[0]['boxes_3d'].tensor.detach().cpu().numpy())
    name = str(time.time())
    save_bbox2bev([ bbox_pts[0]['boxes_3d'][:30] ], img_metas, 'debug_eval', name)
    save_bbox2img(img, [ bbox_pts[0]['boxes_3d'][:30] ], img_metas, 'debug_eval', name)
    ds_name = 'waymo' if len(img_metas[0]['filename'])==5 else 'nuscene'
    # print(img.size())
    # print(img_metas[0])
    for i in range(img.size(0)):
        # img_out = img[i].permute(1,2,0).detach().cpu().numpy()
        img_in = cv2.imread(img_metas[0]['filename'][i])
        cv2.imwrite('debug_eval/{}_input_tensor_{}.png'.format(ds_name, i), img_in)

def save_bbox2bev(gt_bboxes_3d, img_metas, dir = 'debug_coord', name = 'gt_bev'):
    pc = load_pts(img_metas)
    gt_bboxes_3d = gt_bboxes_3d[0]
    reference_points = gt_bboxes_3d.gravity_center.view(1, -1, 3) # 1 num_gt, 3
    save_bev(pc, reference_points, dir, name)

def load_pts(img_metas):
    path = img_metas[0]['pts_filename']
    points = np.fromfile(path, dtype=np.float32)
    dim = 6
    if path.find('waymo') == -1:
        dim=5
    return points.reshape(-1, dim)
    
def save_bev(pts , ref, data_root, out_name = None):
    if isinstance(pts, list):
        pts = pts[0]
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    pc_range= [-75.2, -75.2, -2, 75.2, 75.2, 4]
    mask = ((pts[:, 0] > pc_range[0]) & (pts[:, 0] < pc_range[3]) & 
        (pts[:, 1] > pc_range[1]) & (pts[:, 1] < pc_range[4]) &
        (pts[:, 2] > pc_range[2]) & (pts[:, 2] < pc_range[5]))
    pts = pts[mask]
    res = 0.1
    x_max = 1 + int((pc_range[3] - pc_range[0]) / res)
    y_max = 1 + int((pc_range[4] - pc_range[1]) / res)
    im = torch.zeros(x_max+1, y_max+1, 3)
    x_img = (pts[:, 0] - pc_range[0]) / res
    x_img = x_img.round().long()
    y_img = (pts[:, 1] - pc_range[1]) / res
    y_img = y_img.round().long()
    im[x_img, y_img, :] = 1

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            im[(x_img.long()+i).clamp(min=0, max=x_max), 
                (y_img.long()+j).clamp(min=0, max=y_max), :] = 1
    print('reference', ref.size())
    ref_pts_x = ((ref[..., 0] - pc_range[0]) / res).round().long()
    ref_pts_y = ((ref[..., 1] - pc_range[1]) / res).round().long()
    for i in range(-5,6):
        for j in range(-5,6):
            im[(ref_pts_x.long()+i).clamp(min=0, max=x_max), 
                (ref_pts_y.long()+j).clamp(min=0, max=y_max), 0] = 1
            im[(ref_pts_x.long()+i).clamp(min=0, max=x_max), 
                (ref_pts_y.long()+j).clamp(min=0, max=y_max), 1:2] = 0
    im = im.permute(2, 0, 1)
    timestamp = str(time.time())
    print(timestamp)
    # saved_root = '/home/chenxy/mmdetection3d/'
    if out_name == None:
        out_name = data_root + '/' + timestamp + '.jpg'
    else :
        out_name = data_root + '/' + out_name + '.jpg'
    vutils.save_image(im, out_name)