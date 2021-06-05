from functools import partial
from .losses import *


# class FeatureExtractor(nn.Module):
#     def __init__(self, submodule, cfg, feat):
#         super(FeatureExtractor, self).__init__()
#         self.submodule = submodule
#         self.extracted_layers = []
#         if 'off' not in cfg.distillation['logits']['type']:
#             self.extracted_layers.append(cfg.distillation['logits']['location'])
#         if 'off' not in cfg.distillation['fea']['type']:
#             self.extracted_layers.append(cfg.distillation['fea']['location'])
#         self.feat = feat
#         sub_modules = self.submodule.named_modules()  
#         for name, module in sub_modules:
#             if name in self.extracted_layers:
#                 module.register_forward_hook(partial(self.hook_fn_forward, name=name))

#     def hook_fn_forward(self, module, input, output, name):
#         if name not in self.feat:
#             self.feat[name] = []
#         if self.training == True:
#             self.feat[name].append(output)
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, distillation, feat):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = []
        if 'off' not in distillation['logits']['type']:
            self.extracted_layers.append(distillation['logits']['location'])
        if 'off' not in distillation['fea']['type']:
            self.extracted_layers.append(distillation['fea']['location'])
        self.feat = feat
        sub_modules = submodule.named_modules()  
        for name, module in sub_modules:
            if name in self.extracted_layers:
                module.register_forward_hook(partial(self.hook_fn_forward, name=name))

    def hook_fn_forward(self, module, input, output, name):
        if name not in self.feat:
            self.feat[name] = []
        if self.training == True:
            self.feat[name].append(output)


class DistillationLoss(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, s_cfg, t_cfg):
        super(DistillationLoss, self).__init__()
        # self.kd_loss = CriterionKD()
        self.kd_loss = CriterionKDMSE()
        self.sd_loss = CriterionSDcos_no()
        self.mimic_loss = CriterionMSE()
        self.criterion_fea = VGGFeatureExtractor()
        self.ca_loss = CriterionChannelAwareLoss()
        # self.mimic_loss = nn.ModuleList(
        #     [CriterionMSE(T_channle=t_cfg.MODEL.BACKBONE.OUT_CHANNELS, S_channel=s_cfg.MODEL.BACKBONE.OUT_CHANNELS) for
        #      _ in range(5)])

    def forward(self, soft, pred, distillation, name):
        # import pdb; pdb.set_trace()
        sd_mode = distillation[name]['type']
        lambda_ = distillation[name]['lambda_']
        sd_loss = {}
        if 'KD' in sd_mode:
            loss = 0
            for i in range(len(pred)):
                loss = loss + lambda_['KD'] * self.kd_loss(pred[i], soft[i])
            sd_loss.update({'loss_'+'kd_' + name: loss})

        if 'SD' in sd_mode:
            loss = 0
            for i in range(len(pred)):
                maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0,
                                       ceil_mode=True)
                loss = loss + lambda_['SD'] * self.sd_loss(maxpool(pred[i]), maxpool(soft[i]))
            sd_loss.update({'loss_'+'sd_' + name: loss})

        # if 'mimic' in self.sd_mode:
        #     fea_t = soft
        #     fea_s = pred
        #     loss = 0
        #     for i in range(len(fea_s)):
        #         temp_loss = self.mimic_loss[i](fea_s[i], fea_t[i])
        #         loss = loss + temp_loss
        #     sd_loss.update({'mimic_'+name: loss})
        if 'percep' in sd_mode:
            loss = 0
            for i in range(len(pred)):
                loss = loss + lambda_['percep'] * self.criterion_fea(pred[i], soft[i])
            sd_loss.update({'loss_'+'percep_' + name: loss})

        if 'CA' in sd_mode:
            loss = 0
            for i in range(len(pred)):
                loss = loss + lambda_['CA'] * self.ca_loss(pred[i], soft[i])
            # print('CA_' + name, '|', sd_mode)
            sd_loss.update({'loss_'+'CA_' + name: loss})
        
        return sd_loss
