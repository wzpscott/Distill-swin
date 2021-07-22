import copy
from collections import defaultdict
from itertools import chain

from torch.nn.utils import clip_grad
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.hooks import OptimizerHook

@HOOKS.register_module()
class OptimizerHookGrad(OptimizerHook):
    def after_train_iter(self, runner):
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()
        runner.optimizer.zero_grad()