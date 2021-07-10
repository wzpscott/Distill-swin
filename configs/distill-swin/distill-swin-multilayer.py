_base_ = [
    '../_base_/models/distill-swin.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
log_config = dict(  
    interval=50, 
    hooks=[
        dict(type='TensorboardLoggerHook') 
        # dict(type='TextLoggerHook')
    ])
work_dir = './work_dir/7.9/distill'

model = dict(
    distillation = dict(
        s_patterns=['backbone.layers.[013].blocks.[01].mlp.fc2',
                    'backbone.layers.[2].blocks.[05].mlp.fc2',
                    ],
        t_patterns=['backbone.layers.[013].blocks.[01].mlp.fc2',
                    'backbone.layers.[2].blocks.(0|17).mlp.fc2',
                    ],
        weights_init_strategy='equal',
        parse_mode='regular',
    ),
    s_pretrain = './checkpoints/swin_tiny_patch4_window7_224.pth',
    t_pretrain = './checkpoints/upernet_swin_small_patch4_window7_512x512.pth',
)
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                )

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data = dict(samples_per_gpu=2)
evaluation = dict(interval=2000, metric='mIoU')  

