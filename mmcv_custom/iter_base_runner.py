from mmcv.runner import IterBasedRunner
import os.path as osp
import platform
import shutil
import time
import warnings

import torch
from torch.optim import Optimizer

import mmcv
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.hooks import IterTimerHook
from mmcv.runner.utils import get_host_info

def flatten_grads(grads):
    tmp = [v.flatten() for k,v in grads.items()]
    return torch.cat(tmp)

@RUNNERS.register_module()
class IterBasedRunnerGrad(IterBasedRunner):
    def train(self, data_loader, parse_mode, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.call_hook('before_train_iter')
        if parse_mode == 'regular':
            outputs = self.model.train_step(\
                data_batch, self.optimizer, loss_name = 'all',backward=True,**kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('model.train_step() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter')    
        elif parse_mode == 'SCKD':
            decode_grad = self.model.train_step(\
                data_batch, self.optimizer, loss_name = 'decode.loss_seg',backward=False,**kwargs)
            aux_grad = self.model.train_step(\
                data_batch, self.optimizer, loss_name = 'aux.loss_seg',backward=False,**kwargs)
            soft_grad = self.model.train_step(\
                data_batch, self.optimizer, loss_name = 'loss_decode_head.conv_seg',backward=False,**kwargs)
            loss_name = ['decode.loss_seg']
            if flatten_grads(decode_grad)@flatten_grads(aux_grad) > 0:
                loss_name.append('aux.loss_seg')
            if flatten_grads(decode_grad)@flatten_grads(soft_grad) > 0:
                loss_name.append('loss_decode_head.conv_seg')
            
            outputs = self.model.train_step(\
                data_batch, self.optimizer, loss_name = loss_name,backward=True,**kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('model.train_step() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter') 
        self._inner_iter += 1
        self._iter += 1