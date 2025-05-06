
# _base_ = [
#     '_base_/datasets/partimagenet_clip_640.py',
#     '_base_/default_runtime.py', '_base_/schedules/schedule_20k.py'
# ]

_base_ = [
    '_base_/datasets/tdfm_pin_clip_640.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_20k.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='SimpleDenseCLIP',
    context_length=77,
    text_head=False,
    text_dim=512,
    score_concat_index=2,
    backbone=dict(
        type='CLIPVisionTransformer',
        pretrained='pretrained/ViT-B-16.pt',
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        layers=12,
        input_resolution=640,
        dense=False,
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextEncoder',
        pretrained='pretrained/ViT-B-16.pt',
        context_length=77,  # Default of clip
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    train_cfg=dict(),
    # test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)), 
    test_cfg=dict(mode='whole')
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                warmup_iters=1500,
                warmup_ratio=1e-6)


optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001, 
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.01),
                                        'text_encoder': dict(lr_mult=0.01),  # Not getting trained
                                        'norm': dict(decay_mult=0.)}))

data = dict(samples_per_gpu=4)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')

workflow = [('train', 100), ('val', 2)]