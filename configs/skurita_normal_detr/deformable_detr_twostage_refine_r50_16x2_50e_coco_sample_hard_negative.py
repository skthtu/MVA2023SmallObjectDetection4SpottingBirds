_base_ = 'deformable_detr_twostage_refine_r50_16x2_50e_coco.py'
data_root = 'data/'

data = dict(
    test=dict(
        samples_per_gpu=8,
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_train_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ) 
)

