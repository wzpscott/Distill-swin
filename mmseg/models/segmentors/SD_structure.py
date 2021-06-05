import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.distillation import DistillationLoss
from mmseg.models.distillation.opts import FeatureExtractor
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
import torch


@SEGMENTORS.register_module()
class SDModule(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 cfg, cfg_t,train_cfg,test_cfg,distillation):
        super(SDModule, self).__init__()
        self.teacher = builder.build_segmentor(
            cfg_t, train_cfg=train_cfg, test_cfg=test_cfg)

        state_dict = torch.load(cfg_t.pretrained)['state_dict']
        layers_pretrained = [l for l in state_dict]
        layers = [l for l,_ in self.teacher.named_parameters()]
        print(len(layers_pretrained),len(layers))
        tmp = []
        for l in layers_pretrained:
            if l not in layers:
                tmp.append(l)
        print(len(tmp))
        print(tmp)
        raise ValueError('stop')

        self.teacher.load_state_dict(torch.load(
            cfg_t.pretrained)['state_dict'])

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        
        self.student = builder.build_segmentor(
            cfg, train_cfg=train_cfg, test_cfg=test_cfg)

        self.s_fea = FeatureExtractor(self.student, distillation, feat={})
        self.t_fea = FeatureExtractor(self.teacher, distillation, feat={})

        self.distillation = distillation
        self.cfg_s = cfg
        self.loss = DistillationLoss(s_cfg=cfg, t_cfg=cfg_t)
        self.align_corners = False
        self.test_mode = 'whole'

    def forward_train(self, img, img_metas, gt_semantic_seg):
        loss_dict = self.student(img, img_metas, return_loss=True, gt_semantic_seg=gt_semantic_seg)
        with torch.no_grad():
            _ = self.teacher(img, img_metas, return_loss=True, gt_semantic_seg=gt_semantic_seg)
        del _
        softs_logits = []
        preds_logits = []
        softs_fea = []
        preds_fea = []
        for k in self.t_fea.feat:
            if k in self.distillation['logits']['location']:
                ##apply distillation on the cls logits map
                for ind, soft in enumerate(self.t_fea.feat[k]):
                    softs_logits.append(soft)
                    preds_logits.append(self.s_fea.feat[k][ind])
            elif k in self.distillation['fea']['location']:
                ##apply distillation on the fpn features
                for ind, soft in enumerate(self.t_fea.feat[k]):
                    softs_fea.append(soft)
                    preds_fea.append(self.s_fea.feat[k][ind])
        if "off" not in self.distillation['logits']['type']:
            logits_loss = self.loss(softs_logits, preds_logits, self.distillation, 'logits')
            loss_dict.update(logits_loss)
            del logits_loss
        if "off" not in self.distillation['fea']['type']:
            FEA_loss = self.loss(softs_fea, preds_fea, self.distillation, 'fea')
            loss_dict.update(FEA_loss)
            del FEA_loss
        self.t_fea.feat = {}
        self.s_fea.feat = {}
        return loss_dict

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
