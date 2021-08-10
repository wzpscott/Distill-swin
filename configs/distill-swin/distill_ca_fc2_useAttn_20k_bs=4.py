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
work_dir = './ddp_results/distill_ca_fc2_useAttn_20k_bs=4_2'

model = dict(
        distillation = dict(
        layers=[
                ['backbone.layers.2.blocks.0.mlp.fc2',
                ['backbone.layers.2.blocks.0.mlp.fc2',
                'backbone.layers.2.blocks.1.mlp.fc2',
                'backbone.layers.2.blocks.2.mlp.fc2',
                'backbone.layers.2.blocks.3.mlp.fc2',
                'backbone.layers.2.blocks.4.mlp.fc2'],
                [384,512],
                3],
            ['backbone.layers.2.blocks.1.mlp.fc2',
                ['backbone.layers.2.blocks.3.mlp.fc2',
                'backbone.layers.2.blocks.4.mlp.fc2',
                'backbone.layers.2.blocks.5.mlp.fc2',
                'backbone.layers.2.blocks.6.mlp.fc2',
                'backbone.layers.2.blocks.7.mlp.fc2'],
                [384,512],
                3],
            ['backbone.layers.2.blocks.2.mlp.fc2',
                ['backbone.layers.2.blocks.4.mlp.fc2'
                'backbone.layers.2.blocks.5.mlp.fc2',
                'backbone.layers.2.blocks.6.mlp.fc2',
                'backbone.layers.2.blocks.7.mlp.fc2',
                'backbone.layers.2.blocks.8.mlp.fc2'],
                [384,512],
                3],
            ['backbone.layers.2.blocks.3.mlp.fc2',
                ['backbone.layers.2.blocks.9.mlp.fc2',
                'backbone.layers.2.blocks.10.mlp.fc2',
                'backbone.layers.2.blocks.11.mlp.fc2',
                'backbone.layers.2.blocks.12.mlp.fc2',
                'backbone.layers.2.blocks.13.mlp.fc2'],
                [384,512],
                3],
            ['backbone.layers.2.blocks.4.mlp.fc2',
                ['backbone.layers.2.blocks.12.mlp.fc2',
                'backbone.layers.2.blocks.13.mlp.fc2',
                'backbone.layers.2.blocks.14.mlp.fc2',
                'backbone.layers.2.blocks.15.mlp.fc2',
                'backbone.layers.2.blocks.16.mlp.fc2'],
                [384,512],
                3],
            ['backbone.layers.2.blocks.5.mlp.fc2',
                ['backbone.layers.2.blocks.15.mlp.fc2',
                'backbone.layers.2.blocks.16.mlp.fc2',
                'backbone.layers.2.blocks.17.mlp.fc2'],
                [384,512],
                3],
            # ['decode_head.conv_seg','decode_head.conv_seg',[150,150],2],
        ],
        weights_init_strategy='equal',
        parse_mode='regular',
        use_attn=True
    ),
    s_pretrain = './checkpoints/swin_tiny_patch4_window7_224.pth',
    t_pretrain = './checkpoints/upernet_swin_base_patch4_window7_512x512.pth',
)
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9,0.999), weight_decay=0.01,
                )

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data = dict(samples_per_gpu=4)
evaluation = dict(interval=2000, metric='mIoU')  

runner = dict(type='IterBasedRunnerGrad', max_iters=20000)