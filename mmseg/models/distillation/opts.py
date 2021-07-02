from functools import partial
from .losses import *
import re
from collections import OrderedDict
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
    def __init__(self, submodule, patterns):
        super(Extractor, self).__init__()
        sub_modules_names = [name for name,_ in submodule.named_modules()]
        self.extracted_layers = []
        for name in sub_modules_names:
            for pat in patterns:
                if re.match(pat,name):
                    self.extracted_layers.append(name)
                    break

        self.feat = []
        self.shape = []
        self.layers = []

        sub_modules = submodule.named_modules()
        for name, module in sub_modules:
            if name in self.extracted_layers:
                if 'fc' in name:
                    out_dim = module.out_features
                else:
                    out_dim = 16
                self.shape.append(out_dim)
                self.layers.append(name)
                module.register_forward_hook(partial(self.hook_fn_forward, name=name))

    def hook_fn_forward(self, module, input, output, name):
        if self.training == True:
            self.feat.append(output)


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
    def __init__(self,input_size,output_size):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Conv1d(input_size,input_size, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv1d(output_size,output_size, kernel_size=1, stride=1, padding=0),
        )
    def forward(self,x):
        return self.ff(x)

class DistillationLoss_(nn.Module):
    def __init__(self, s_cfg, t_cfg,s_shapes,t_shapes,distillation,layers):
        super().__init__()
        self.kd_loss = CriterionKDMSE()

        self.adaptors = nn.ModuleList()
        self.s_norms = nn.ModuleList()
        self.t_norms = nn.ModuleList()

        for i in range(len(s_shapes)):
            s_shape = s_shapes[i]
            t_shape = t_shapes[i]

            self.s_norms.append(nn.BatchNorm1d(t_shape))
            self.t_norms.append(nn.BatchNorm1d(t_shape))

            if s_shape  == t_shape:
                self.adaptors.append(None)
            else:
                self.adaptors.append(Adaptor(s_shape,t_shape))

        self.layers = layers
        
        # add gradients to weight of each layer's loss
        self.strategy = distillation['weights_init_strategy']
        if self.strategy=='equal':
            # weights = [5*1e3 for i in range(s_shape)]
            weights = [1e6,1e6,1e6,1e6,1e5,1e5,1e3,1e3]
            weights = nn.Parameter(torch.Tensor(weights),requires_grad=False)
            self.weights = weights
        elif self.strategy=='weight_average':
            self.sd_weight = nn.Parameter(torch.Tensor([0.8]),requires_grad=True)
        else:
            raise ValueError('Wrong weights init strategy')

    def forward(self, soft, pred, losses):
        for i in range(len(pred)):
            adaptor=self.adaptors[i]
            pred[i] = pred[i].permute(0,2,1)
            soft[i] = soft[i].permute(0,2,1)

            if adaptor is None:
                pred[i] = pred[i]
            else:
                pred[i] = adaptor(pred[i])
            
            pred[i] = self.s_norms[i](pred[i])
            soft[i] = self.t_norms[i](soft[i])
            
            pred[i] = pred[i].permute(0,2,1)
            soft[i] = soft[i].permute(0,2,1)

            # maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0,
            #                         ceil_mode=True)
            # loss = self.weights[i]*self.kd_loss(maxpool(pred[i]), maxpool(soft[i]))

            loss = self.weights[i]*self.kd_loss(pred[i], soft[i])
            name = self.layers[i]
            losses.update({'loss_'+name: loss})

        return losses
