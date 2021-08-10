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
    def __init__(self, student, teacher, layers):
        super().__init__()

        self.teacher_features = []
        self.student_features = []
        self.channel_dims = []  # student 和 teacher 被提取层的输出通道数
        self.total_dims = []  # student 和 teacher 被提取层的输出维数

        for i,(student_layer,teacher_layer,channel_dim,total_dim) in enumerate(layers):
            self.channel_dims.append(channel_dim)
            self.total_dims.append(total_dim)

            if not isinstance(teacher_layer,list):
                teacher_layer = [teacher_layer]

            for name, module in teacher.named_modules():
                if name in teacher_layer:
                    module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='teacher',layer_num=i))
                    print(f'teacher_layer :{teacher_layer} hooked!!!!')

            for name, module in student.named_modules():
                if name == student_layer:
                    module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='student'))
                    print(f'student_layer :{student_layer} hooked!!!!')


    def hook_fn_forward(self, module, input, output, name, type,layer_num=None):
        if self.training == True:
            if type == 'student':
                self.student_features.append(output)
            if type == 'teacher':
                if len(self.teacher_features)>layer_num:
                    self.teacher_features[layer_num].append(output)
                else:
                    self.teacher_features.append([output])

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
class ff(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        # self.ff = nn.Sequential(
        #     nn.Conv1d(input_size,input_size, kernel_size=1, stride=1, padding=0),
        #     nn.GELU(),
        #     nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0),
        #     nn.GELU(),
        #     nn.Conv1d(output_size,output_size, kernel_size=1, stride=1, padding=0),
        # )
        
        self.ff = nn.Sequential(
            nn.Linear(input_size,input_size),
            nn.GELU(),
            nn.Linear(input_size,output_size),
            nn.GELU(),
            nn.Linear(output_size,output_size),
        )
    def forward(self,x):
        x = self.ff(x)
        return x

# class AttnAdaptor(nn.Module):
#     def __init__(self,input_size,output_size,teacher_num):
#         super().__init__()
#         Ws = nn.ModuleList()
#         for _ in range(teacher_num):
#             Ws.append(ff(input_size,output_size))
#         self.Ws = Ws

#         self.W = ff(input_size,output_size)
#     def forward(self,x_student,x_teachers):
#         # x_student:[b,WH,C_s]
#         # x_teachers:List of tensor([b,WH,C_t])
#         T = 1
#         b,WH,C = x_teachers[0].shape

#         x_students = []
#         for i,x_teacher in enumerate(x_teachers):
#             # x_student_ = x_student.permute(0,2,1)
#             # x_student_ = self.Ws[i](x_student_)
#             # x_student_ = x_student_.permute(0,2,1)
#             x_student_ = self.Ws[i](x_student)
#             x_students.append(x_student_)

#         x_students = torch.stack(x_students,dim=0) # x_students:[teacher_num,b,WH,C_t]
#         x_teachers = torch.stack(x_teachers,dim=0) # x_teachers:[teacher_num.b,WH,C_t]

#         x_teachers_ = x_teachers.reshape(-1,b*WH*C) # x_teachers:[teacher_num,b*WH*C]
#         x_students_ = x_students.reshape(-1,b*WH*C)# x_student:[teacher_num,b*WH*C]

#         t_mag = torch.sqrt(torch.sum(x_teachers_*x_teachers_,dim=1))
#         s_mag = torch.sqrt(torch.sum(x_students_*x_students_,dim=1))

#         attn = torch.sum((x_teachers_*x_students_),dim=1)/(t_mag*s_mag)
#         attn = F.softmax(attn/T) # [teacher_num]

#         x_teachers = torch.mul(x_teachers, attn.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
        
#         # x_student = x_student.permute(0,2,1)
#         x_student = self.W(x_student)
#         # x_student = x_student.permute(0,2,1)

#         return x_student,x_teachers,attn
class conv1d(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.conv = nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0)
    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = x.permute(0,2,1)
        return x

class AttnAdaptor(nn.Module):
    def __init__(self,student_size,teacher_size,teacher_num):
        super().__init__()
        Wt = nn.ModuleList()
        for _ in range(teacher_num):
            Wt.append(conv1d(teacher_size,1))
        self.Wt = Wt
        self.Ws = conv1d(student_size,1)
    def forward(self,x_student,x_teachers):
        # T = 4
        # b,WH,C_t = x_teachers[0].shape
        # teacher_num = len(x_teachers)
        # C_s = x_student.shape[2]

        # # print(teacher_num)
        # # print(x_teachers[0].shape,x_student.shape)
        # # print('---------------------')

        # # x_teachers = torch.stack(x_teachers,dim=0) # x_teachers:[teacher_num,b,WH,C_t]
        # # x_teachers = x_teachers.permute(0,3,2,1).reshape(teacher_num,WH,b*C_t) # x_teachers:[teacher_num,WH,b*C_t]
        # # # x_teachers = F.softmax(x_teachers,dim=1).mean(dim=2) # x_teachers:[teacher_num,WH]
        # # # x_teachers = x_teachers.mean(dim=2) # x_teachers:[teacher_num,WH]
        
        # for i in range(len(x_teachers)):
        #     x_teachers[i] = self.Wt[i](x_teachers[i]) # x_teachers[i]:[b,WH,1]
        #     x_teachers[i] = x_teachers[i].squeeze(2) # x_teachers[i]:[b,WH]
        # x_teachers = torch.stack(x_teachers,dim=0) # x_teachers:[teacher_num,b,WH]

        # # print(x_teachers.shape)

        # x_student = self.Ws(x_student).squeeze(2) # x_student:[b,WH]
        # x_student = x_student.unsqueeze(0) # x_student:[1,b,WH]

        # # print(x_student.shape)
        # # print('---------------------')

        # # x_student = x_student.permute(1,0,2).reshape(WH,b*C_s) # x_student:[WH,b*C_s]
        # # x_student = F.softmax(x_student,dim=0).mean(dim=1).unsqueeze(0) # x_student:[1,WH]
        # # x_student = x_student.mean(dim=1).unsqueeze(0) # x_student:[1,WH]
        
        # attn = (x_teachers*x_student).sum(dim=2) # attn:[teacher_num,b]
        # # print('attn_before',attn)
        # attn = F.softmax(attn/T,dim=0).unsqueeze(2) # attn:[teacher_num,b,1]
        # # print('attn:',attn)
        # # print(attn.shape)
        # # print('---------------------')

        # x_teacher = (attn*x_teachers).sum(dim=0).unsqueeze(0) # x_teacher:[1,b,WH]
        # # print('x_teacher',x_teacher)
        # # print(x_teacher.shape,x_student.shape)
        # return x_student,x_teacher,attn.mean(dim=1)

        T = 1
        b, WH, C_t = x_teachers[0].shape
        teacher_num = len(x_teachers)
        C_s = x_student.shape[2]

        # x_teachers = torch.stack(x_teachers,dim=0) # x_teachers:[teacher_num,b,WH,C_t]
        # x_teachers = x_teachers.permute(0,3,2,1).reshape(teacher_num,WH,b*C_t) # x_teachers:[teacher_num,WH,b*C_t]
        # # x_teachers = F.softmax(x_teachers,dim=1).mean(dim=2) # x_teachers:[teacher_num,WH]
        # # x_teachers = x_teachers.mean(dim=2) # x_teachers:[teacher_num,WH]

        for i in range(len(x_teachers)):
            x_teachers[i] = self.Wt[i](x_teachers[i])  # x_teachers[i]:[b,WH,1]
            x_teachers[i] = x_teachers[i].squeeze(2)  # x_teachers[i]:[b,WH]
        x_teachers = torch.stack(x_teachers, dim=0)  # x_teachers:[teacher_num,b,WH]

        x_student = self.Ws(x_student).squeeze(2)  # x_student:[b,WH]
        x_student = x_student.unsqueeze(0)  # x_student:[1,b,WH]

        # x_student = x_student.permute(1,0,2).reshape(WH,b*C_s) # x_student:[WH,b*C_s]
        # x_student = F.softmax(x_student,dim=0).mean(dim=1).unsqueeze(0) # x_student:[1,WH]
        # x_student = x_student.mean(dim=1).unsqueeze(0) # x_student:[1,WH]

        attn = (x_teachers * x_student).sum(dim=2)  # attn:[teacher_num,b]
        # print('attn_before',attn)
        attn = F.softmax(attn / T, dim=0).unsqueeze(2)  # attn:[teacher_num,b,1]
        # print('attn:',attn)

        x_teacher = (attn * x_teachers).sum(dim=0).unsqueeze(0)  # x_teacher:[b,1,WH]
        # x_student = x_student.permute(1,0,2)
        # print('x_teacher',x_teacher)
        return x_student, x_teacher, attn.mean(dim=1)

class Adaptor(nn.Module):
    def __init__(self,input_size,output_size,total_dim):
        super().__init__()
        self.total_dim = total_dim

        ff = nn.Sequential(
            nn.Conv1d(input_size,input_size, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv1d(output_size,output_size, kernel_size=1, stride=1, padding=0),
        )

        if total_dim == 3:
            # self.ff = nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0)
            self.ff = ff
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
    def __init__(self,distillation,tau):
        super().__init__()
        # self.kd_loss = CriterionChannelAwareLoss2D(tau)
        self.kd_loss = nn.MSELoss()
        self.adaptors = nn.ModuleList()

        layers = distillation['layers']
        self.use_attn = distillation['use_attn']
        for _,teacher_layers,channel_dim,_ in layers:
            student_dim,teacher_dim = channel_dim
            # self.adaptors.append(Adaptor(student_dim,teacher_dim,total_dim))
            if self.use_attn:
                self.adaptors.append(AttnAdaptor(student_dim,teacher_dim,len(teacher_layers)))
            else:
                self.adaptors.append(Adaptor(student_dim,teacher_dim,3))
            print(f'add an adaptor of shape {student_dim} to {teacher_dim}')

        self.layers = [student_name for student_name,_,_,_ in layers]
        
        # add gradients to weight of each layer's loss
        self.strategy = distillation['weights_init_strategy']
        if self.strategy=='equal':
            # weights = [1e8,1e7,1e6,1e4,1e3]
            weights = [1 for i in range(len(layers))]
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

            if self.use_attn:
                pred[i],soft[i],attn = adaptor(pred[i],soft[i])

                for j in range(attn.shape[0]):
                    losses.update({'attn'+str(i)+'layer'+str(j): attn[j].clone()})
            else:
                soft[i] = soft[i][0]
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

# from functools import partial
# from .losses import *
# import re
# from collections import OrderedDict

# import torch
# import torch.nn as nn 
# import torch.nn.functional as F 

# class FeatureExtractor(nn.Module):
#     def __init__(self, submodule, distillation, feat):
#         super(FeatureExtractor, self).__init__()
#         self.submodule = submodule
#         self.extracted_layers = []
#         if 'off' not in distillation['logits']['type']:
#             self.extracted_layers.append(distillation['logits']['location'])
#         if 'off' not in distillation['fea']['type']:
#             self.extracted_layers.append(distillation['fea']['location'])
#         self.feat = feat
#         sub_modules = submodule.named_modules()  
#         for name, module in sub_modules:
#             if name in self.extracted_layers:
#                 module.register_forward_hook(partial(self.hook_fn_forward, name=name))

#     def hook_fn_forward(self, module, input, output, name):
#         if name not in self.feat:
#             self.feat[name] = []
#         if self.training == True:
#             self.feat[name].append(output)


# class Extractor(nn.Module):
#     def __init__(self, student, teacher, layers):
#         super().__init__()

#         self.teacher_features = []
#         self.student_features = []
#         self.channel_dims = []  # student 和 teacher 被提取层的输出通道数
#         self.total_dims = []  # student 和 teacher 被提取层的输出维数

#         for i,(student_layer,teacher_layer,channel_dim,total_dim) in enumerate(layers):
#             self.channel_dims.append(channel_dim)
#             self.total_dims.append(total_dim)

#             if not isinstance(teacher_layer,list):
#                 teacher_layer = [teacher_layer]

#             for name, module in teacher.named_modules():
#                 if name in teacher_layer:
#                     module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='teacher',layer_num=i))
#                     print(f'teacher_layer :{teacher_layer} hooked!!!!')

#             for name, module in student.named_modules():
#                 if name == student_layer:
#                     module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='student'))
#                     print(f'student_layer :{student_layer} hooked!!!!')


#     def hook_fn_forward(self, module, input, output, name, type,layer_num=None):
#         if self.training == True:
#             if type == 'student':
#                 self.student_features.append(output)
#             if type == 'teacher':
#                 if len(self.teacher_features)>layer_num:
#                     self.teacher_features[layer_num].append(output)
#                 else:
#                     self.teacher_features.append([output])

# class DistillationLoss(nn.Module):
#     """
#     Main class for Generalized R-CNN. Currently supports boxes and masks.
#     It consists of three main parts:
#     - backbone
#     - rpn
#     - heads: takes the features + the proposals from the RPN and computes
#         detections / masks from it.
#     """

#     def __init__(self, s_cfg, t_cfg):
#         super().__init__()
#         # self.kd_loss = CriterionKD()
#         self.kd_loss = CriterionKDMSE()
#         self.sd_loss = CriterionSDcos_no()
#         self.mimic_loss = CriterionMSE()
#         self.criterion_fea = VGGFeatureExtractor()
#         self.ca_loss = CriterionChannelAwareLoss()
#         # self.mimic_loss = nn.ModuleList(
#         #     [CriterionMSE(T_channle=t_cfg.MODEL.BACKBONE.OUT_CHANNELS, S_channel=s_cfg.MODEL.BACKBONE.OUT_CHANNELS) for
#         #      _ in range(5)])

#     def forward(self, soft, pred, distillation, name):
#         # import pdb; pdb.set_trace()
#         sd_mode = distillation[name]['type']
#         lambda_ = distillation[name]['lambda_']
#         sd_loss = {}
#         if 'KD' in sd_mode:
#             loss = 0
#             for i in range(len(pred)):
#                 loss = loss + lambda_['KD'] * self.kd_loss(pred[i], soft[i])
#             sd_loss.update({'loss_'+'kd_' + name: loss})

#         if 'SD' in sd_mode:
#             loss = 0
#             for i in range(len(pred)):
#                 maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0,
#                                        ceil_mode=True)
#                 loss = loss + lambda_['SD'] * self.sd_loss(maxpool(pred[i]), maxpool(soft[i]))
#             sd_loss.update({'loss_'+'sd_' + name: loss})

#         # if 'mimic' in self.sd_mode:
#         #     fea_t = soft
#         #     fea_s = pred
#         #     loss = 0
#         #     for i in range(len(fea_s)):
#         #         temp_loss = self.mimic_loss[i](fea_s[i], fea_t[i])
#         #         loss = loss + temp_loss
#         #     sd_loss.update({'mimic_'+name: loss})
#         if 'percep' in sd_mode:
#             loss = 0
#             for i in range(len(pred)):
#                 loss = loss + lambda_['percep'] * self.criterion_fea(pred[i], soft[i])
#             sd_loss.update({'loss_'+'percep_' + name: loss})

#         if 'CA' in sd_mode:
#             loss = 0
#             print(pred[0].shape,soft[0].shape)
#             for i in range(len(pred)):
#                 loss = loss + lambda_['CA'] * self.ca_loss(pred[i], soft[i])
#             print(self.ca_loss(pred[i], soft[i]))
#             sd_loss.update({'loss_'+'CA_' + name: loss})
        
#         return sd_loss


# class conv1d(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         self.conv = nn.Conv1d(input_size, output_size, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = self.conv(x)
#         x = x.permute(0, 2, 1)
#         return x


# class AttnAdaptor(nn.Module):
#     def __init__(self, student_size, teacher_size, teacher_num):
#         super().__init__()
#         Wt = nn.ModuleList()
#         for _ in range(teacher_num):
#             Wt.append(conv1d(teacher_size, 1))
#         self.Wt = Wt
#         self.Ws = conv1d(student_size, 1)

#     def forward(self, x_student, x_teachers):
#         T = 1
#         b, WH, C_t = x_teachers[0].shape
#         teacher_num = len(x_teachers)
#         C_s = x_student.shape[2]

#         # x_teachers = torch.stack(x_teachers,dim=0) # x_teachers:[teacher_num,b,WH,C_t]
#         # x_teachers = x_teachers.permute(0,3,2,1).reshape(teacher_num,WH,b*C_t) # x_teachers:[teacher_num,WH,b*C_t]
#         # # x_teachers = F.softmax(x_teachers,dim=1).mean(dim=2) # x_teachers:[teacher_num,WH]
#         # # x_teachers = x_teachers.mean(dim=2) # x_teachers:[teacher_num,WH]

#         for i in range(len(x_teachers)):
#             x_teachers[i] = self.Wt[i](x_teachers[i])  # x_teachers[i]:[b,WH,1]
#             x_teachers[i] = x_teachers[i].squeeze(2)  # x_teachers[i]:[b,WH]
#         x_teachers = torch.stack(x_teachers, dim=0)  # x_teachers:[teacher_num,b,WH]

#         x_student = self.Ws(x_student).squeeze(2)  # x_student:[b,WH]
#         x_student = x_student.unsqueeze(0)  # x_student:[1,b,WH]

#         # x_student = x_student.permute(1,0,2).reshape(WH,b*C_s) # x_student:[WH,b*C_s]
#         # x_student = F.softmax(x_student,dim=0).mean(dim=1).unsqueeze(0) # x_student:[1,WH]
#         # x_student = x_student.mean(dim=1).unsqueeze(0) # x_student:[1,WH]

#         attn = (x_teachers * x_student).sum(dim=2)  # attn:[teacher_num,b]
#         # print('attn_before',attn)
#         attn = F.softmax(attn / T, dim=0).unsqueeze(2)  # attn:[teacher_num,b,1]
#         # print('attn:',attn)

#         x_teacher = (attn * x_teachers).sum(dim=0).unsqueeze(0)  # x_teacher:[b,1,WH]
#         # x_student = x_student.permute(1,0,2)
#         # print('x_teacher',x_teacher)
#         return x_student, x_teacher, attn.mean(dim=1)
# class Adaptor(nn.Module):
#     def __init__(self,input_size,output_size,total_dim):
#         super().__init__()
#         self.total_dim = total_dim

#         # self.ff = nn.Sequential(
#         #     nn.Conv1d(input_size,input_size, kernel_size=1, stride=1, padding=0),
#         #     nn.GELU(),
#         #     nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0),
#         #     nn.GELU(),
#         #     nn.Conv1d(output_size,output_size, kernel_size=1, stride=1, padding=0),
#         # )
#         if total_dim == 3:
#             self.ff = nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0)
#         elif total_dim == 4:
#             self.ff = nn.Conv2d(input_size,output_size, kernel_size=1, stride=1, padding=0)
#     def forward(self,x):
#         if self.total_dim == 2:
#             x = x
#         elif self.total_dim == 3:
#             x = x.permute(0,2,1)
#             x = self.ff(x)
#             x = x.permute(0,2,1)
#         elif self.total_dim == 4:
#             x = self.ff(x)
#         else:
#             raise ValueError('wrong total_dim')
#         return x

# class DistillationLoss_(nn.Module):
#     def __init__(self, distillation, tau):
#         super().__init__()

#         self.adaptors = nn.ModuleList()

#         layers = distillation['layers']
#         self.use_attn = distillation['use_attn']
#         if self.use_attn:
#             self.kd_loss = CriterionChannelAwareLoss2D(tau=tau)
#         else:
#             self.kd_loss = nn.MSELoss()
#         for _,teacher_layers,channel_dim,total_dim in layers:
#             student_dim,teacher_dim = channel_dim
#             if self.use_attn:
#                 self.adaptors.append(AttnAdaptor(student_dim,teacher_dim,len(teacher_layers)))
#             else:
#                 self.adaptors.append(Adaptor(student_dim,teacher_dim,total_dim))
#             print(f'add an adaptor of shape {student_dim} to {teacher_dim} of total dim {total_dim}')

#         self.layers = [student_name for student_name,_,_,_ in layers]
        
#         # add gradients to weight of each layer's loss
#         self.strategy = distillation['weights_init_strategy']
#         if self.strategy=='equal':
#             # weights = [1e8,1e7,1e6,1e4,1e3]
#             weights = [1 for i in range(len(layers))]
#             # weights = [1,1,1,1,1,1]
#             weights = nn.Parameter(torch.Tensor(weights),requires_grad=False)
#             self.weights = weights
#         elif self.strategy=='self_adjust':
#             weights = nn.Parameter(torch.Tensor([1 for i in range(len(layers))]),requires_grad=True)
#             self.weights = weights
#         else:
#             raise ValueError('Wrong weights init strategy')

#     def forward(self, soft, pred, losses):
#         for i in range(len(pred)):
#             adaptor=self.adaptors[i]

#             if self.use_attn:
#                 pred[i],soft[i],attn = adaptor(pred[i],soft[i])

#                 for j in range(attn.shape[0]):
#                     losses.update({'attn'+str(i)+'layer'+str(j): attn[j].clone()})
#             else:
#                 soft[i] = soft[i][0]
#                 pred[i] = adaptor(pred[i])


#             if self.strategy=='equal':
#                 loss = self.weights[i]*self.kd_loss(pred[i], soft[i])
#                 name = self.layers[i]
#                 losses.update({'loss_'+name: loss})
#             elif self.strategy == 'self_adjust':
#                 loss = (1/(self.weights[i]**2))*\
#                         self.kd_loss(pred[i], soft[i])\
#                         +torch.log(self.weights[i])
#                 name = self.layers[i]
#                 losses.update({'loss_'+name: loss})
#                 losses.update({'weight_'+name: self.weights[i]})
            
             
#         # if self.strategy=='equal':
#         #     pass
#         # elif self.strategy=='self_adjust':
#         #     losses['decode.loss_seg'] =(1/(self.weights[1]**2))*losses['decode.loss_seg']+torch.log(self.weights[1])
#         #     losses['aux.loss_seg'] = (1/(self.weights[2]**2))*losses['aux.loss_seg']+torch.log(self.weights[2])
#         #
#         #     losses.update({'weight_'+'decode.loss_seg': self.weights[1]})
#         #     losses.update({'weight_'+'aux.loss_seg': self.weights[2]})
#         return losses

