from set_lib_dir import LIB_ROOT_DIR

dataset_type = 'DroneDataset'  
data_root = LIB_ROOT_DIR + '/data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize',
        img_scale= (3264,1836),
        multiscale_mode='value',
        override=True,
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
   dict(
        type='MultiScaleFlipAug',
        img_scale=(3264,1836),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(3264,1836), keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'drone2021/annotations/split_train_coco.json',
        img_prefix=data_root + 'drone2021/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'drone2021/annotations/split_val_coco.json',
        img_prefix=data_root + 'drone2021/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'drone2021/annotations/split_val_coco.json',
        img_prefix=data_root + 'drone2021/images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
