import cv2
import numpy as np
import time
import torchvision.utils as vutils
import torch
import copy
import os
import torchvision.transforms as Trans
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import random

def visualize_gt(gt_bboxes_3d=None, gt_labels_3d=None, img_metas = None, name = None, dir_name = 'debug', gtvis_range = [0,105]):#labels not utilized
    if name == None:
        name = str(time.time())
    # score_type = str(round(float(test_output[0]['scores_3d'][j]),3))+'||type:'+str(int(test_output[0]['labels_3d'][j]))
    centers = gt_bboxes_3d[0].gravity_center
    mask = torch.zeros(centers.shape[0])
    for i in range(centers.shape[0]):
        dist = float(torch.sqrt(centers[i,0]**2+centers[i,1]**2))
        mask[i] = (gtvis_range[0]<dist and dist<gtvis_range[1])
    gt_bboxes = [gt_bboxes_3d[0][mask==1]]
    gt_labels = [gt_labels_3d[0][mask==1]]
    gt = [{'scores_3d':np.ones(len(gt_bboxes[0])), 'labels_3d': gt_labels[0]}]
    save_bbox2bev( gt_bboxes, img_metas, dir_name, name, gt)
    save_bbox2img( None, gt_bboxes, img_metas, dir_name, name, gt)

def save_bbox_pred(bbox_pts, img, img_metas, name = None , dir_name = 'debug', vis_count = None):
    print(len(bbox_pts[0]))
    if name == None:
        name = str(time.time())
    if vis_count == None:
        for i,conf in enumerate(bbox_pts[0]['scores_3d']):
            if conf < 0.3:
                break
        vis_count = max(i,30)
    save_bbox2bev([ bbox_pts[0]['boxes_3d'][:vis_count] ], img_metas, dir_name, name+f'_top{vis_count}', bbox_pts)
    save_bbox2img(img, [ bbox_pts[0]['boxes_3d'][:vis_count] ], img_metas, dir_name, name+f'_top{vis_count}', bbox_pts)

def save_bbox2img(img, gt_bboxes_3d, img_metas, dirname='debug_coord', name = None, test_output = None):
    # ds_name = 'waymo' if len(img_metas[0]['filename'])==5 else 'nuscene'
    gt_bboxes_3d = gt_bboxes_3d[0]
    reference_points = gt_bboxes_3d.gravity_center.view(1, -1, 3) # 1 num_gt, 3
    print(reference_points.size())
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    ref_3d = reference_points.clone()
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)##B num_c num_q 4 ,1
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)   # B num_c num_q 4 4
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)    # B num_c num_q 4
    ref_img = reference_points_cam.clone()
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)   #filter out negative depth, B num_c num_q
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(  #z for depth, too shallow will cause zero division
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)    # eps controls minimum
    ref_pixel = reference_points_cam.clone()
    if type(img_metas[0]['ori_shape']) == tuple:    
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
    # print(mask.any()==False)
    if (name == None):
        name = str(time.time())
    for i in range(num_cam):
        img_out = cv2.imread(img_metas[0]['filename'][i])
        h,w,_ = img_metas[0]['ori_shape'][0]#img_out.shape
        print(h,w)

        for j in range(num_query):
            pt = reference_points_cam[0,i,j]
            # print(pt,'  ',mask[0,i,j])
            if mask[0,i,j] == True:
                color = np.random.randint(256, size=3)
                color = [int(x) for x in color]
                cv2.circle(img_out, (int(pt[0]*w),int(pt[1]*h)), radius=6,color=color, thickness = 4)
                #  cv2.circle(imgs[j], (int(pt[0]),int(pt[1])), radius=5 , color = color, thickness = 4)
                if test_output!=None:
                    score_type = str(round(float(test_output[0]['scores_3d'][j]),3))+'Cls'+str(int(test_output[0]['labels_3d'][j]))
                    cv2.putText(img_out, score_type, 
                        (int(pt[0]*w),int(pt[1]*h)),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=color)
        cv2.imwrite(dirname+'/{}_{}.png'.format(name, i), img_out)

def save_bbox2bev(gt_bboxes_3d, img_metas, dir = 'debug_coord', name = 'gt_bev', test_output = None):
    pc = load_pts(img_metas)
    gt_bboxes_3d = gt_bboxes_3d[0]
    reference_points = gt_bboxes_3d.gravity_center.view(1, -1, 3) # 1 num_gt, 3
    save_bev(pc, reference_points, dir, name, test_output)

def load_pts(img_metas):
    path = img_metas[0]['pts_filename']
    points = np.fromfile(path, dtype=np.float32)
    dim = 6
    if path.find('waymo') == -1:
        dim=5
    return points.reshape(-1, dim)

def save_bev(pts , ref, data_root, out_name = None, test_output = None):
    if isinstance(pts, list):
        pts = pts[0]
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    pc_range= [-75.2, -75.2, -2, 75.2, 75.2, 4]
    mask = ((pts[:, 0] > pc_range[0]) & (pts[:, 0] < pc_range[3]) & 
        (pts[:, 1] > pc_range[1]) & (pts[:, 1] < pc_range[4]) &
        (pts[:, 2] > pc_range[2]) & (pts[:, 2] < pc_range[5]))
    pts = pts[mask]
    res = 0.05
    x_max = 1 + int((pc_range[3] - pc_range[0]) / res)
    y_max = 1 + int((pc_range[4] - pc_range[1]) / res)
    im = torch.zeros(x_max+1, y_max+1, 3)
    x_img = (pts[:, 0] - pc_range[0]) / res
    x_img = x_img.round().long()
    y_img = (pts[:, 1] - pc_range[1]) / res
    y_img = y_img.round().long()
    im[x_img, y_img, :] = 1

    for i in [ 0]:
        for j in [0]:
            im[(x_img.long()+i).clamp(min=0, max=x_max), 
                (y_img.long()+j).clamp(min=0, max=y_max), :] = 1
    print('reference', ref.size())
    ref_pts_x = ((ref[..., 0] - pc_range[0]) / res).round().long()
    ref_pts_y = ((ref[..., 1] - pc_range[1]) / res).round().long()
    for i in range(-2,3):
        for j in range(-2,3):
            im[(ref_pts_x.long()+i).clamp(min=0, max=x_max), 
                (ref_pts_y.long()+j).clamp(min=0, max=y_max), 0] = 1
            im[(ref_pts_x.long()+i).clamp(min=0, max=x_max), 
                (ref_pts_y.long()+j).clamp(min=0, max=y_max), 1:2] = 0
    im = im.permute(2, 0, 1)
    timestamp = str(time.time())
    print(timestamp)
    # saved_root = '/home/chenxy/mmdetection3d/'
    if out_name == None:
        out_name = data_root + '/' + timestamp + '_bev.jpg'
    else :
        out_name = data_root + '/' + out_name + '_bev.jpg'
    T = Trans.ToPILImage()
    img = T(im)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('/usr/share/fonts/gnu-free/FreeMonoBold.ttf',24)
    num_q = ref_pts_x.shape[-1]
    for i in range(num_q):
        x,y = int(ref_pts_y[0,i]), int(ref_pts_x[0,i])
        if 'gt' in out_name:
            r=10
        else:
            r=10
        dy = 0#random.randint(0,20)*30
        if i % 10 ==0:
            color = tuple([int(x) for x in np.random.randint(256, size=3)])
        if test_output!=None:
            score_type = str(int(test_output[0]['labels_3d'][i])) + str(round(float(test_output[0]['scores_3d'][i]),2))[1:]
            draw.text((x+r-3,y+r-3) ,score_type , color, font=font)
        draw.arc([(x-r,y-r),(x+r,y+r)],0,360, width=100, fill=color)
        # draw.line([x,y,x,y+dy], fill=color, width=3)
    img.save(out_name)
    # vutils.save_image(im, out_name)