_base_ = [
    # '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]

plugin=True
plugin_dir='projects/mmdet3d_plugin/'
# load_from = 'ckpts/fcos3d_yue.pth'
# load_from = 'ckpts/waymo_pretrain_pgd_mv_8gpu_for_detr3d_backbone_statedict_only.pth'
# resume_from = '/home/zhenglt/pure-detr3d/work_dirs/waymo_nuscenes/epoch_1.pth'

point_cloud_range = [-51, -75, -2, 75, 75, 4]
voxel_size = [0.5, 0.5, 6]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
load_from='/home/zhenglt/pure-detr3d/ckpts/fcos3d_yue.pth'
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# voxel_size = [0.2, 0.2, 8]
# img_norm_cfg = dict(
#     mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
nus_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian'
]
waymo_class_names = [
    'Car', 'Pedestrian', 'Cyclist'
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='Detr3D',
    use_grid_mask=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        # with_cp=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='Detr3DHead',
        num_query=900,
        num_classes=3,
        in_channels=256,
        code_size=8,    #we don't infer velocity here, but infer(x,y,z,w,h,l,sin(θ),cos(θ)) for bboxes
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], #specify the weights since default length is 10
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='Detr3DTransformer',
            num_cams = 6,
            decoder=dict(
                type='Detr3DTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='Detr3DCrossAtten',
                            pc_range=point_cloud_range,
                            num_cams = 6,
                            waymo_with_nuscene = True,#align cams   
                            num_points=1,
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=point_cloud_range,
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=3), 
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range))))

nus_dataset = 'NuScenesDataset'
nus_data_root = 'data/nuscenes/'
waymo_dataset = 'CustomWaymoDataset'
# waymo_data_root = 'data/waymo_v131/kitti_format/'
waymo_data_root = '/localdata_ssd/waymo_ssd_train_only/kitti_format/'
file_client_args = dict(backend='disk')

nus_train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # dict(type='MyResize',img_scale=1/4), #unknown error: import failed
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=nus_class_names),
    dict(type='ProjectLabelToWaymoClass', class_names = nus_class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=nus_class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
waymo_img_scale = (1066, 1600)
waymo_train_pipeline = [
    dict(type='MyLoadMultiViewImageFromFiles', to_float32=True, img_scale=(1280, 1920)),#do paddings for ill-shape imgs
    dict(type='MyResize', img_scale=waymo_img_scale, keep_ratio=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    # dict(type='MyFilterBoxOutofImage'),
    dict(type='MyLoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=waymo_class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=waymo_class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
waymo_test_pipeline = [
    dict(type='MyLoadMultiViewImageFromFiles', to_float32=True, img_scale=(1280, 1920)),#original scale
    dict(type='MyResize', img_scale=waymo_img_scale, keep_ratio=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale = waymo_img_scale, #(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=waymo_class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ])
]

nus_train = dict(
        type=nus_dataset,
        data_root=nus_data_root,
        ann_file=nus_data_root + 'nuscenes_infos_train.pkl',
        pipeline=nus_train_pipeline,
        classes=nus_class_names,
        with_velocity=False,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR')

waymo_train = dict(
        type=waymo_dataset,
        data_root=waymo_data_root,
        num_views=5,
        ann_file=waymo_data_root + 'waymo_infos_train.pkl',
        split='training',
        pipeline=waymo_train_pipeline,
        modality=input_modality,
        classes=waymo_class_names,
        test_mode=False,
        box_type_3d='LiDAR',
        load_interval=5)
waymo_val = dict(
        type=waymo_dataset,
        data_root=waymo_data_root,
        num_views=5,
        ann_file=waymo_data_root + 'waymo_infos_val.pkl',
        split='training',
        pipeline=waymo_test_pipeline,
        modality=input_modality,
        classes=waymo_class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        load_interval=5)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train = [nus_train,waymo_train],
    # train = waymo_train,
    val = waymo_val, 
    test = waymo_val)

optimizer = dict(
    type='AdamW', 
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

