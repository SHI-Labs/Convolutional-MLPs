_base_ = [
    './_base_/models/fpn_r50.py',
    './_base_/datasets/ade20k.py',
    './_base_/default_runtime.py'
]

model = dict(
    pretrained='http://ix.cs.uoregon.edu/~alih/conv-mlp/checkpoints/convmlp_s_imagenet.pth',
    backbone=dict(type='SegConvMLPSmall'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150))

# optimizer
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict()

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')
