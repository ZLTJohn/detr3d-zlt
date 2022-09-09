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
from .zltvis import save_temporal_frame
@DATASETS.register_module()
class CustomWaymoDataset_T(KittiDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Waymo Dataset.

    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': box in LiDAR coordinates
            - 'Depth': box in depth coordinates, usually for indoor dataset
            - 'Camera': box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [-85, -85, -5, 85, 85, 5].
    """

    CLASSES = ('Car', 'Pedestrian', 'Sign', 'Cyclist')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 num_views=5,
                 history_len = 1, skip_len = 0,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5]):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range)

        self.num_views = num_views
        assert self.num_views <= 5
        self.history_len = history_len
        self.skip_len = skip_len
        self.load_interval = load_interval
        # to load a subset, just set the load_interval in the dataset config
        self.data_infos_full = self.data_infos
        self.data_infos = self.data_infos[::load_interval]
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]

    def _get_pts_filename(self, idx):
        pts_filename = osp.join(self.root_split, self.pts_prefix,
                                f'{idx:07d}.bin')
        return pts_filename

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        # import time
        # _ = time.time()
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            # print('dataloading cost: {} ms'.format(time.time()-_))
            return data

    def prepare_train_data(self, index):
        #[T, T-1]
        index = index * self.load_interval
        idx_list = list(range(index-self.history_len, index))
        random.shuffle(idx_list)
        idx_list = idx_list[self.skip_len:] + [index]#skip frame
        idx_list = sorted(idx_list, reverse=True)
        data_queue = []
        scene_id = None
        for i in idx_list:
            i = max(0,i)
            input_dict = self.get_data_info(i)
            if scene_id == None: scene_id = input_dict['sample_idx']//1000
            if input_dict is None: return None
            if scene_id == input_dict['sample_idx']//1000:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
            data_queue.insert(0, copy.deepcopy(example))
        if self.filter_empty_gt and\
            (data_queue[-1] is None or ~(data_queue[-1]['gt_labels_3d']._data != -1).any()):
            return None
        #data_queue: T-len+1, T
        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        input: queue: dict of [T-len+1, T], containing data_info
        convert sample queue into one single sample.
        calculate transformation from ego_now to image_old
        note that we dont gather gt objects of previous frames
        """
        # oldname='queue'
        # np.save('debug/debug_temporal1/'+oldname, queue)
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        ego2global = queue[-1]['img_metas'].data['pose']
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            global2ego_old = np.linalg.inv(metas_map[i]['pose'])
            ego2img_old_rts = []
            for ego_old2img_old in metas_map[i]['lidar2img']:
                ego2img_old =ego_old2img_old @ global2ego_old @ ego2global #@pt_ego
                ego2img_old_rts.append(ego2img_old)
            metas_map[i]['lidar2img'] = ego2img_old_rts
        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        # breakpoint()
        # save_temporal_frame(queue)
        # name = 'queue_union'
        # np.save('debug/debug_temporal1/'+name, queue)
        # breakpoint()
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Standard input_dict consists of the
                data information.

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str | None): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        # index=475  # in infos_train.pkl is index 485
        if self.test_mode == True:
            info = self.data_infos[index]
        else: info = self.data_infos_full[index]
        
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        lidar2img = P0 @ rect @ Trv2c

        # the Tr_velo_to_cam is computed for all images but not saved in .info for img1-4
        # the size of img0-2: 1280x1920; img3-4: 886x1920
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []

            # load calibration for all 5 images.
            calib_path = img_filename.replace('image_0', 'calib').replace('.png', '.txt')
            Tr_velo_to_cam_list = []
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            for line_num in range(6, 6 + self.num_views):
                trans = np.array([float(info) for info in lines[line_num].split(' ')[1:13]]).reshape(3, 4)
                trans = np.concatenate([trans, np.array([[0., 0., 0., 1.]])], axis=0).astype(np.float32)
                Tr_velo_to_cam_list.append(trans)
            assert np.allclose(Tr_velo_to_cam_list[0], info['calib']['Tr_velo_to_cam'].astype(np.float32))

            for idx_img in range(self.num_views):
                rect = info['calib']['R0_rect'].astype(np.float32)
                # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
                Trv2c = Tr_velo_to_cam_list[idx_img]
                P0 = info['calib'][f'P{idx_img}'].astype(np.float32)
                lidar2img = P0 @ rect @ Trv2c

                image_paths.append(img_filename.replace('image_0', f'image_{idx_img}'))
                lidar2img_rts.append(lidar2img)

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
        )
        if self.modality['use_camera']:
            input_dict['img_filename'] = image_paths
            input_dict['lidar2img'] = lidar2img_rts
        input_dict['pose'] = info['pose']

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        
        return input_dict

    def get_ann_info(self, index):
        # Use index to get the annos, thus the evalhook could also use this api
        if self.test_mode == True:
            info = self.data_infos[index]
        else: info = self.data_infos_full[index]
        
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)

        # convert gt_bboxes_3d to velodyne coordinates
        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ Trv2c))
        gt_bboxes = annos['bbox']

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names)
        return anns_results

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None,
                       data_format='waymo'):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            data_format (str | None): Output data format. Default: 'waymo'.
                Another supported choice is 'kitti'.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        assert ('waymo' in data_format or 'kitti' in data_format), \
            f'invalid data_format {data_format}'
        print("still work before format_results ---  if not isinstance")
        # np.save('debug_eval/zltwaymo_eval_result_before_format_results',outputs)
        # print('saved!')
        # exit(0)
        if (not isinstance(outputs[0], dict)) or 'img_bbox' in outputs[00]:
            raise TypeError('Not supported type for reformat results.')
        elif 'pts_bbox' in outputs[0]:#we go this way
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name#saving path
                if submission_prefix is not None:
                    submission_prefix_ = f'{submission_prefix}_{name}'
                else:
                    submission_prefix_ = None
                result_files_ = self.bbox2result_kitti(results_, self.CLASSES,
                                                       pklfile_prefix_,
                                                       submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES,
                                                  pklfile_prefix,
                                                  submission_prefix)
        # print(result_files)
        # np.save('debug_eval/zltwaymo_eval_result_kitti_format',result_files)## turn into cam-coord, it sucks
        # exit(0)
        # open('zlt_output_kitti_format_debug.txt','w').write(str(result_files))  #we got absolutely right data
        # exit(0)  
        if 'waymo' in data_format:
            from .zlt_kitti2waymo import zlt_KITTI2Waymo as KITTI2Waymo
            waymo_root = osp.join(
                self.data_root.split('kitti_format')[0], 'waymo_format')
            if self.split == 'training':
                waymo_tfrecords_dir = osp.join(waymo_root, 'validation')
                prefix = '1'
            elif self.split == 'testing':
                waymo_tfrecords_dir = osp.join(waymo_root, 'testing')
                prefix = '2'
            else:
                raise ValueError('Not supported split value.')
            save_tmp_dir = tempfile.TemporaryDirectory()
            waymo_results_save_dir = save_tmp_dir.name
            waymo_results_final_path = f'{pklfile_prefix}.bin'
            print("still work before converter init!!!")
            if 'pts_bbox' in result_files:#result_files deprecated
                converter = KITTI2Waymo(result_files['pts_bbox'],
                                        waymo_tfrecords_dir,
                                        waymo_results_save_dir,
                                        waymo_results_final_path, prefix)
            else:
                converter = KITTI2Waymo(result_files, waymo_tfrecords_dir,
                                        waymo_results_save_dir,
                                        waymo_results_final_path, prefix)# worker弄小一点看看行不行
            print("still work before converter convert!!!")
            print(waymo_tfrecords_dir, waymo_results_save_dir, waymo_results_final_path)
            # exit(0)
            converter.convert()         
            print("still work after converter convert!!!")
            save_tmp_dir.cleanup()

        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='waymo',
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default: 'waymo'. Another supported metric is 'kitti'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.

        Returns:
            dict[str: float]: results of each evaluation metric
        """
        print("metric here is-----------{}".format(metric))
        # np.save('debug_eval/zltwaymo_eval_result',results)
        # print('saved!')## result still correct here!
        # exit(0)
        assert ('waymo' in metric or 'kitti' in metric), \
            f'invalid metric {metric}'

        if 'waymo' in metric:
            waymo_root = osp.join(
                self.data_root.split('kitti_format')[0], 'waymo_format')
            if pklfile_prefix is None:
                eval_tmp_dir = tempfile.TemporaryDirectory()
                pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
            else:
                eval_tmp_dir = None
            result_files, tmp_dir = self.format_results(
                results,
                pklfile_prefix,
                submission_prefix,
                data_format='waymo')# xxxxxxx not found inside, maybe it's OK

            import shutil
            shutil.copy(f'{pklfile_prefix}.bin', 'work_dirs/result.bin')
            from time import time
            _ = time()
            ap_dict = compute_waymo_let_metric(f'{waymo_root}/gt.bin', f'{pklfile_prefix}.bin')
            print('time usage of compute_let_metric: {} s'.format(time()-_))        
            
            if eval_tmp_dir is not None:
                eval_tmp_dir.cleanup()

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir)
        return ap_dict

    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert results to kitti format for evaluation and test submission.

        Args:
            net_outputs (List[np.ndarray]): list of array storing the
                bbox and score
            class_nanes (List[String]): A list of class names
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            List[dict]: A list of dict have the kitti 3d format
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)
        # np.save('debug_eval/zltwaymo_eval_net_outputs',net_outputs)   # data_size * [output one frame]
        # exit(0)
        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]     # 所以我们output是stack起来的，哦因为你eval肯定不需要做random shuffle啊。。。idx直接能对上，所以output不用存
            sample_idx = info['image']['image_idx']
            image_shape = info['image']['image_shape'][:2]
            # if you are going to replace final result.bin with gt boxes, do it here
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            # np.save('debug_eval/zltwaymo_box_dict',box_dict)
            # print(box_dict)
            # exit(0)
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                anno = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(
                        -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

                if submission_prefix is not None:
                    curr_file = f'{submission_prefix}/{sample_idx:07d}.txt'
                    with open(curr_file, 'w') as f:
                        bbox = anno['bbox']
                        loc = anno['location']
                        dims = anno['dimensions']  # lhw -> hwl

                        for idx in range(len(bbox)):
                            print(
                                '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.
                                format(anno['name'][idx], anno['alpha'][idx],
                                       bbox[idx][0], bbox[idx][1],
                                       bbox[idx][2], bbox[idx][3],
                                       dims[idx][1], dims[idx][2],
                                       dims[idx][0], loc[idx][0], loc[idx][1],
                                       loc[idx][2], anno['rotation_y'][idx],
                                       anno['score'][idx]),
                                file=f)
            else:
                annos.append({
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                })
            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the boxes into valid format.

        Args:
            box_dict (dict): Bounding boxes to be converted.

                - boxes_3d (:obj:``LiDARInstance3DBoxes``): 3D bounding boxes.
                - scores_3d (np.ndarray): Scores of predicted boxes.
                - labels_3d (np.ndarray): Class labels of predicted boxes.
            info (dict): Dataset information dictionary.

        Returns:
            dict: Valid boxes after conversion.

                - bbox (np.ndarray): 2D bounding boxes (in camera 0).
                - box3d_camera (np.ndarray): 3D boxes in camera coordinates.
                - box3d_lidar (np.ndarray): 3D boxes in lidar coordinates.
                - scores (np.ndarray): Scores of predicted boxes.
                - label_preds (np.ndarray): Class labels of predicted boxes.
                - sample_idx (np.ndarray): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        # TODO: remove the hack of yaw
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        P0 = box_preds.tensor.new_tensor(P0)    # that is to say, box_2d_pred only projected to cam0 image！

        box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c) #box3d in camera coord

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P0)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        valid_inds = valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )