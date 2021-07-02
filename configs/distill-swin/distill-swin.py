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
work_dir = './work_dir/CA=1_layers.2.blocks.5.mlp/'

# model = dict(
#     cfg=dict(
#         pretrained='./checkpoints/swin_tiny_patch4_window7_224.pth',
        
#     ),
#     cfg_t=dict(
#         pretrained='./checkpoints/upernet_swin_base_patch4_window7_512x512.pth',
#     ),
#     distillation = dict(
#         logits=dict(type='CA', location='backbone.layers.2.blocks.5.mlp.fc2', lambda_=dict(KD=0, SD=0.0, CA=1)),
#         fea=dict(type='SD', location='decode_head.bottleneck.activate', lambda_=dict(KD=0, SD=0)),
#         mask=dict()
#     ),
# )


"""
distillation format:
distillation = {
    location_list = [layer1,layer2,...],
}
"""
model = dict(
    distillation = dict(
        logits=dict(type='CA', location='backbone.layers.2.blocks.5.mlp.fc2', lambda_=dict(KD=0, SD=0.0, CA=1)),
        fea=dict(type='SD', location='decode_head.bottleneck.activate', lambda_=dict(KD=0, SD=0)),
        mask=dict()
    ),
    s_pretrain = './checkpoints/swin_tiny_patch4_window7_224.pth',
    t_pretrain = './checkpoints/upernet_swin_base_patch4_window7_512x512.pth',
)
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                )

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
evaluation = dict(interval=2000, metric='mIoU')

data=dict(samples_per_gpu=8)