from functools import partial
from .losses import *
import re
from collections import OrderedDict

import torch
import torch.nn as nn 
import torch.nn.functional as F 

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


class Extractor(nn.Module):
    def __init__(self, teacher, student, layers):
        super().__init__()

        self.teacher_features = []
        self.student_features = []
        self.channel_dims = []  # student 和 teacher 被提取层的输出通道数
        self.total_dims = []  # student 和 teacher 被提取层的输出维数

        for student_layer,teacher_layer,channel_dim,total_dim in layers:
            self.channel_dims.append(channel_dim)
            self.total_dims.append(total_dim)

            for name, module in teacher.named_modules():
                if name == teacher_layer:
                    module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='teacher'))
                    print(f'teacher_layer :{teacher_layer} hooked!!!!')

            for name, module in student.named_modules():
                if name == student_layer:
                    module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='student'))
                    print(f'student_layer :{student_layer} hooked!!!!')


    def hook_fn_forward(self, module, input, output, name, type):
        if self.training == True:
            if type == 'student':
                self.student_features.append(output)
            if type == 'teacher':
                self.teacher_features.append(output)

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
        super().__init__()
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
            print(pred[0].shape,soft[0].shape)
            for i in range(len(pred)):
                loss = loss + lambda_['CA'] * self.ca_loss(pred[i], soft[i])
            print(self.ca_loss(pred[i], soft[i]))
            sd_loss.update({'loss_'+'CA_' + name: loss})
        
        return sd_loss

class Adaptor(nn.Module):
    def __init__(self,input_size,output_size,total_dim):
        super().__init__()
        self.total_dim = total_dim

        # self.ff = nn.Sequential(
        #     nn.Conv1d(input_size,input_size, kernel_size=1, stride=1, padding=0),
        #     nn.GELU(),
        #     nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0),
        #     nn.GELU(),
        #     nn.Conv1d(output_size,output_size, kernel_size=1, stride=1, padding=0),
        # )
        if total_dim == 3:
            self.ff = nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0)
        elif total_dim == 4:
            self.ff = nn.Conv2d(input_size,output_size, kernel_size=1, stride=1, padding=0)
    def forward(self,x):
        if self.total_dim == 2:
            x = x
        elif self.total_dim == 3:
            x = x.permute(0,2,1)
            x = self.ff(x)
            x = x.permute(0,2,1)
        elif self.total_dim == 4:
            x = self.ff(x)
        else:
            raise ValueError('wrong total_dim')
        return x

class DistillationLoss_(nn.Module):
    def __init__(self,distillation):
        super().__init__()
        self.kd_loss = CriterionChannelAwareLoss()

        self.adaptors = nn.ModuleList()

        layers = distillation['layers']
        for _,_,channel_dim,total_dim in layers:
            student_dim,teacher_dim = channel_dim
            self.adaptors.append(Adaptor(student_dim,teacher_dim,total_dim))
            print(f'add an adaptor of shape {student_dim} to {teacher_dim}')

        self.layers = [student_name for student_name,_,_,_ in layers]
        
        # add gradients to weight of each layer's loss
        self.strategy = distillation['weights_init_strategy']
        if self.strategy=='equal':
            # weights = [1e8,1e7,1e6,1e4,1e3]
            weights = [1]
            # weights = [1,1,1,1,1,1]
            weights = nn.Parameter(torch.Tensor(weights),requires_grad=False)
            self.weights = weights
        elif self.strategy=='self_adjust':
            weights = nn.Parameter(torch.Tensor([1 for i in range(3)]),requires_grad=True)
            self.weights = weights
        else:
            raise ValueError('Wrong weights init strategy')

    def forward(self, soft, pred, losses):
        for i in range(len(pred)):
            adaptor=self.adaptors[i]
            pred[i] = adaptor(pred[i])

            if self.strategy=='equal':
                loss = self.weights[i]*self.kd_loss(pred[i], soft[i])
                name = self.layers[i]
                losses.update({'loss_'+name: loss})
            elif self.strategy=='self_adjust':
                loss = (1/(self.weights[0]**2))*\
                        self.kd_loss(pred[i], soft[i])\
                        +torch.log(self.weights[0])
                name = self.layers[i]
                losses.update({'loss_'+name: loss})
                losses.update({'weight_'+name: self.weights[0]})
        if self.strategy=='equal':
            pass
        elif self.strategy=='self_adjust':
            losses['decode.loss_seg'] =(1/(self.weights[1]**2))*losses['decode.loss_seg']+torch.log(self.weights[1])
            losses['aux.loss_seg'] = (1/(self.weights[2]**2))*losses['aux.loss_seg']+torch.log(self.weights[2])

            losses.update({'weight_'+'decode.loss_seg': self.weights[1]})
            losses.update({'weight_'+'aux.loss_seg': self.weights[2]})
        return losses
