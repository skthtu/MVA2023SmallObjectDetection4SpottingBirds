from set_lib_dir import LIB_ROOT_DIR
import os
_base_ = './deformable_detr_twostage_refine_r50_16x2_50e_coco.py'
data_root = LIB_ROOT_DIR + '/data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadHardNegatives'),
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


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        hard_negative_file=LIB_ROOT_DIR + '/work_dirs/deformable_detr_twostage_refine_r50_16x2_50e_coco/train_coco_hard_negative.json',  # ---
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_train_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    val=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    test=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    )
)


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[15, 18])
runner = dict(max_epochs=20)
load_from = LIB_ROOT_DIR + '/work_dirs/deformable_detr_twostage_refine_r50_16x2_50e_coco_finetune/latest.pth'
