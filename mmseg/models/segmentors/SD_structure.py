import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmseg.models.distillation import DistillationLoss
from mmseg.models.distillation.opts import FeatureExtractor,Extractor,DistillationLoss_
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
import torch
from collections import OrderedDict
import numpy as np

@SEGMENTORS.register_module()
class SDModule_(BaseSegmentor):
    def __init__(self,
                 cfg, cfg_t,train_cfg,test_cfg,distillation,s_pretrain,t_pretrain):
        super(SDModule_, self).__init__()
        self.cfg_s = cfg
        self.cfg_t = cfg_t
        self.teacher = builder.build_segmentor(
            cfg_t, train_cfg=train_cfg, test_cfg=test_cfg)
        self.teacher.load_state_dict(torch.load(
            t_pretrain)['state_dict'])

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.student = builder.build_segmentor(
            cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        self.student_init(strategy='use_pretrain',s_pretrain=s_pretrain,t_pretrain=t_pretrain)

        self.s_fea = Extractor(self.student, distillation.s_patterns)
        self.t_fea = Extractor(self.teacher, distillation.t_patterns)

        self.layers = self.s_fea.extracted_layers

        self.distillation = distillation
        
        self.loss = DistillationLoss_(s_cfg=cfg, t_cfg=cfg_t,
                                    s_shapes=self.s_fea.shape,t_shapes=self.t_fea.shape,
                                    distillation = distillation,layers=self.layers,
                                    )
        self.align_corners = False
        self.test_mode = 'whole'

        self.counter = {'total':0,'aux.loss_seg':0,'loss_backbone.layers.0.blocks.0.mlp.fc2':0,
        'loss_backbone.layers.0.blocks.1.mlp.fc2':0,'loss_backbone.layers.1.blocks.0.mlp.fc2':0,
         'loss_backbone.layers.1.blocks.1.mlp.fc2':0, 'loss_backbone.layers.2.blocks.0.mlp.fc2':0,
         'loss_backbone.layers.2.blocks.5.mlp.fc2':0,'loss_backbone.layers.3.blocks.0.mlp.fc2':0,
         'loss_backbone.layers.3.blocks.1.mlp.fc2':0
        }
        

    def forward_train(self, img, img_metas, gt_semantic_seg):
        loss_dict = self.student(img, img_metas, return_loss=True, gt_semantic_seg=gt_semantic_seg)
        with torch.no_grad():
            _ = self.teacher(img, img_metas, return_loss=True, gt_semantic_seg=gt_semantic_seg)
        del _


        softs_fea = []
        preds_fea = []

        for i in range(len(self.s_fea.feat)):
            soft = self.t_fea.feat[i]
            pred = self.s_fea.feat[i]

            softs_fea.append(soft)
            preds_fea.append(pred)

        loss_dict = self.loss(softs_fea, preds_fea, loss_dict)

        self.t_fea.feat = []
        self.s_fea.feat = []

        return loss_dict
    
    def student_init(self,strategy,s_pretrain=None,t_pretrain=None,distillation=None):
        if strategy == 'use_pretrain':# 使用预训练模型
            # 载入student的权重
            # 预训练模型的层名称没有‘backbone.’ 的前缀，因此在载入前需要增加前缀
            state_dict = torch.load(s_pretrain)['model']
            new_keys = ['backbone.'+key for key in state_dict]
            d1 = dict( zip( list(state_dict.keys()), new_keys) )
            new_state_dict = {d1[oldK]: value for oldK, value in state_dict.items()}
            self.student.load_state_dict(new_state_dict,strict=False)
        elif strategy == 'use_teacher' :# 跳层初始化
            assert self.cfg_s['backbone']['embed_dim'] == self.cfg_t['backbone']['embed_dim']  # 需要维度一致

            state_dict = torch.load(t_pretrain)['state_dict']  # 载入teacher模型
            # student 和 teacher对应: 0->0 1->3 2->6 3->10 4->14 5->17
            mapping = {0:0,1:3,2:6,3:10,4:14,5:17}
            new_state_dict = OrderedDict()
            # print([k for k,v in state_dict.items()])
            for k,v in state_dict.items():
                if not k.startswith('backbone.layers.2'):
                    new_state_dict[k] = v
                elif str(k.split('.')[4]) in mapping.keys():
                    new_k = k.split('.')
                    new_k[4] = mapping[new_k[4]]
                    new_k = ''.join(new_k)

                    new_state_dict[new_k] = v
                    
            self.student.load_state_dict(new_state_dict,strict=False)


        else:
            raise ValueError('Wrong student init strategy')

    def get_grad(self,loss):
        loss.backward(retain_graph=True)
        grads = torch.Tensor([]).cuda()
        for name,param in self.student.named_parameters():
            if param.grad is None:
                grad = torch.zeros(param.shape).cuda()
                grads = torch.cat([grads,grad.flatten()])
            else:
                grads = torch.cat([grads,param.grad.flatten()])
        self.student.zero_grad()
        return grads
    # @staticmethod
    def _parse_losses(self,losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = 0
        decode_grad = self.get_grad(log_vars['decode.loss_seg'])
        for key,value in log_vars.items():
            if 'loss' in key and 'decode' not in key:
                loss_grad = self.get_grad(log_vars[key])
                cos = loss_grad@decode_grad
                if cos>0:
                    loss += value
                else:
                    self.counter[key] += 1
                self.counter['total'] += 1
        if (self.counter['total']+1) % 10 == 0:
            print(self.counter)
            print({k:v/self.counter['total'] for k,v in self.counter.items()})
        del decode_grad
        # loss = sum(_value for _key, _value in log_vars.items()
        #            if 'loss' in _key) 
                   
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap."""

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                pad_img = crop_img.new_zeros(
                    (crop_img.size(0), crop_img.size(1), h_crop, w_crop))
                pad_img[:, :, :y2 - y1, :x2 - x1] = crop_img
                pad_seg_logit = self.s_net.encode_decode(pad_img, img_meta)
                preds[:, :, y1:y2,
                x1:x2] += pad_seg_logit[:, :, :y2 - y1, :x2 - x1]
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.student.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        flip_direction = img_meta[0]['flip_direction']
        if flip:
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred