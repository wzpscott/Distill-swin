_base_ = [
    '../_base_/models/distill-swin.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
log_config = dict(  
    interval=50, 
    hooks=[
        dict(type='TensorboardLoggerHook') 
        # dict(type='TextLoggerHook')
    ])
work_dir = './work_dir/6.4/KD=0,SD=0/'

model = dict(
    cfg=dict(
        backbone=dict(
            use_checkpoint='./checkpoints/swin_tiny_patch4_window7_224.pth',
        )
        
    ),
    cfg_t=dict(
        pretrained='./checkpoints/upernet_swin_base_patch4_window7_512x512.pth',
    ),
    distillation = dict(
        logits=dict(type='CA', location='decode_head.conv_seg', lambda_=dict(KD=0.0, SD=0.0, CA=0.0)),
        fea=dict(type='SD', location='decode_head.bottleneck.activate', lambda_=dict(KD=0, SD=0)),
        mask=dict()
    ),
)

# optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.)}))
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                )

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data=dict(samples_per_gpu=8)