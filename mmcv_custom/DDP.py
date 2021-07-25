import torch
from torch.nn.parallel.distributed import (DistributedDataParallel,
                                           _find_tensors)

from mmcv.utils import TORCH_VERSION
from mmcv.parallel.scatter_gather import scatter_kwargs

import deepspeed
import sys
import types

from packaging import version as pkg_version

from deepspeed import ops
from deepspeed import module_inject

from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.engine import ADAM_OPTIMIZER, LAMB_OPTIMIZER
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.inference.engine import InferenceEngine

from deepspeed.runtime.lr_schedules import add_tuning_arguments
from deepspeed.runtime.config import DeepSpeedConfig, DeepSpeedConfigError
from deepspeed.runtime.activation_checkpointing import checkpointing
from deepspeed.ops.transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig
from deepspeed.module_inject import replace_transformer_layer, revert_transformer_layer

from deepspeed.utils import log_dist
from deepspeed.utils.distributed import init_distributed

from deepspeed.runtime import zero

from deepspeed.pipe import PipelineModule

from deepspeed.git_version_info import version, git_hash, git_branch


class MyDistributedDataParallel(DistributedDataParallel):
     def forward(self, *inputs, **kwargs):
        if self.ddp_uneven_inputs_config.ddp_join_enabled:
            ones = torch.ones(
                1, device=self.device
            )
            work = dist.all_reduce(ones, group=self.process_group, async_op=True)
            self.reducer._set_forward_pass_work_handle(
                work, self.ddp_uneven_inputs_config.ddp_join_divide_by_initial_world_size
            )

        # Calling _rebuild_buckets before forward compuation,
        # It may allocate new buckets before deallocating old buckets
        # inside _rebuild_buckets. To save peak memory usage,
        # call _rebuild_buckets before the peak memory usage increases
        # during forward computation.
        # This should be called only once during whole training period.
        
        # if self.reducer._rebuild_buckets():
        #     logging.info("Reducer buckets have been rebuilt in this iteration.")

        if self.require_forward_param_sync:
            self._sync_params()

        if self.ddp_uneven_inputs_config.ddp_join_enabled:
            # Notify joined ranks whether they should sync in backwards pass or not.
            self._check_global_requires_backward_grad_sync(is_joined_rank=False)

        if self.device_ids:
            if len(self.device_ids) == 1:
                inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
                output = self.module(*inputs[0], **kwargs[0])
            else:
                inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
                outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module(*inputs, **kwargs)

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False

        return output

class MMDistributedDataParallel(MyDistributedDataParallel):
    """The DDP module that supports DataContainer.

    MMDDP has two main differences with PyTorch DDP:

    - It supports a custom type :class:`DataContainer` which allows more
      flexible control of input data.
    - It implement two APIs ``train_step()`` and ``val_step()``.
    """

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def train_step(self, *inputs, **kwargs):
        """train_step() API for module wrapped by DistributedDataParallel.

        This method is basically the same as
        ``DistributedDataParallel.forward()``, while replacing
        ``self.module.forward()`` with ``self.module.train_step()``.
        It is compatible with PyTorch 1.1 - 1.5.
        """
        if getattr(self, 'require_forward_param_sync', True):
            self._sync_params()
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.train_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.train_step(*inputs, **kwargs)

        if torch.is_grad_enabled() and getattr(
                self, 'require_backward_grad_sync', True):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            if TORCH_VERSION > '1.2':
                self.require_forward_param_sync = False
        return output

    def val_step(self, *inputs, **kwargs):
        """val_step() API for module wrapped by DistributedDataParallel.

        This method is basically the same as
        ``DistributedDataParallel.forward()``, while replacing
        ``self.module.forward()`` with ``self.module.val_step()``.
        It is compatible with PyTorch 1.1 - 1.5.
        """
        if getattr(self, 'require_forward_param_sync', True):
            self._sync_params()
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.val_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.val_step(*inputs, **kwargs)

        if torch.is_grad_enabled() and getattr(
                self, 'require_backward_grad_sync', True):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            if TORCH_VERSION > '1.2':
                self.require_forward_param_sync = False
        return output


class MMDeepSpeedEngine(DeepSpeedEngine):
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
    def train_step(self, *inputs, **kwargs):
        r"""Execute forward propagation

        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """
        if self.flops_profiler_enabled(
        ) and self.global_steps == self.flops_profiler_profile_step(
        ) and self.global_rank == 0:
            self.flops_profiler.start_profile(ignore_list=None)

        if self.module.training and self.progressive_layer_drop:
            kwargs.update(self.progressive_layer_drop.get_state())

        if self.zero_optimization_partition_weights():
            # Enable automated discovery of external parameters by indicating that
            # we are in a forward pass.
            for module in self.module.modules():
                module._parameters._in_forward = True
                pass

        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
            self.timers('forward').start()

        if self.training_dataloader is None:
            self.tput_timer.start()
        loss = self.module.train_step(*inputs, **kwargs)

        if self.zero_optimization_partition_weights():
            # Reset the ZeRO-3 state if we are only doing forward-passes (ie evaluation).
            if not torch._C.is_grad_enabled():
                self.optimizer.param_coordinator.reset_step()

            # Disable automated discovery of external parameters
            for module in self.module.modules():
                module._parameters._in_forward = False

        if self.wall_clock_breakdown():
            self.timers('forward').stop()
            self.timers('forward_microstep').stop()

        if self.flops_profiler_enabled(
        ) and self.global_steps == self.flops_profiler_profile_step(
        ) and self.global_rank == 0:
            self.flops_profiler.stop_profile()
            self.flops_profiler.print_model_profile(
                profile_step=self.global_steps,
                module_depth=self.flops_profiler_module_depth(),
                top_modules=self.flops_profiler_top_modules(),
                detailed=self.flops_profiler_detailed(),
                output_file=self.flops_profiler_output_file())
            self.flops_profiler.end_profile()
        return loss
    def val_step(self, *inputs, **kwargs):
        r"""Execute forward propagation

        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """
        if self.flops_profiler_enabled(
        ) and self.global_steps == self.flops_profiler_profile_step(
        ) and self.global_rank == 0:
            self.flops_profiler.start_profile(ignore_list=None)

        if self.module.training and self.progressive_layer_drop:
            kwargs.update(self.progressive_layer_drop.get_state())

        if self.zero_optimization_partition_weights():
            # Enable automated discovery of external parameters by indicating that
            # we are in a forward pass.
            for module in self.module.modules():
                module._parameters._in_forward = True
                pass

        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
            self.timers('forward').start()

        if self.training_dataloader is None:
            self.tput_timer.start()

        loss = self.module.val_step(*inputs, **kwargs)

        if self.zero_optimization_partition_weights():
            # Reset the ZeRO-3 state if we are only doing forward-passes (ie evaluation).
            if not torch._C.is_grad_enabled():
                self.optimizer.param_coordinator.reset_step()

            # Disable automated discovery of external parameters
            for module in self.module.modules():
                module._parameters._in_forward = False

        if self.wall_clock_breakdown():
            self.timers('forward').stop()
            self.timers('forward_microstep').stop()

        if self.flops_profiler_enabled(
        ) and self.global_steps == self.flops_profiler_profile_step(
        ) and self.global_rank == 0:
            self.flops_profiler.stop_profile()
            self.flops_profiler.print_model_profile(
                profile_step=self.global_steps,
                module_depth=self.flops_profiler_module_depth(),
                top_modules=self.flops_profiler_top_modules(),
                detailed=self.flops_profiler_detailed(),
                output_file=self.flops_profiler_output_file())
            self.flops_profiler.end_profile()
        return loss




def _parse_version(version_str):
    '''Parse a version string and extract the major, minor, and patch versions.'''
    ver = pkg_version.parse(version_str)
    return ver.major, ver.minor, ver.micro
# Export version information
__version__ = version
__version_major__, __version_minor__, __version_patch__ = _parse_version(__version__)
__git_hash__ = git_hash
__git_branch__ = git_branch

def MMinitialize(args=None,
               model=None,
               optimizer=None,
               model_parameters=None,
               training_data=None,
               lr_scheduler=None,
               mpu=None,
               dist_init_required=None,
               collate_fn=None,
               config=None,
               config_params=None):
    """Initialize the DeepSpeed Engine.

    Arguments:
        args: an object containing local_rank and deepspeed_config fields.
            This is optional if `config` is passed.

        model: Required: nn.module class before apply any wrappers

        optimizer: Optional: a user defined optimizer, this is typically used instead of defining
            an optimizer in the DeepSpeed json config.

        model_parameters: Optional: An iterable of torch.Tensors or dicts.
            Specifies what Tensors should be optimized.

        training_data: Optional: Dataset of type torch.utils.data.Dataset

        lr_scheduler: Optional: Learning Rate Scheduler Object. It should define a get_lr(),
            step(), state_dict(), and load_state_dict() methods

        mpu: Optional: A model parallelism unit object that implements
            get_{model,data}_parallel_{rank,group,world_size}()

        dist_init_required: Optional: None will auto-initialize torch.distributed if needed,
            otherwise the user can force it to be initialized or not via boolean.

        collate_fn: Optional: Merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.

        config: Optional: Instead of requiring args.deepspeed_config you can pass your deepspeed config
            as an argument instead, as a path or a dictionary.

        config_params: Optional: Same as `config`, kept for backwards compatibility.

    Returns:
        A tuple of ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``

        * ``engine``: DeepSpeed runtime engine which wraps the client model for distributed training.

        * ``optimizer``: Wrapped optimizer if a user defined ``optimizer`` is supplied, or if
          optimizer is specified in json config else ``None``.

        * ``training_dataloader``: DeepSpeed dataloader if ``training_data`` was supplied,
          otherwise ``None``.

        * ``lr_scheduler``: Wrapped lr scheduler if user ``lr_scheduler`` is passed, or
          if ``lr_scheduler`` specified in JSON configuration. Otherwise ``None``.
    """
    log_dist("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(
        __version__,
        __git_hash__,
        __git_branch__),
             ranks=[0])

    assert model is not None, "deepspeed.initialize requires a model"

    if not isinstance(model, PipelineModule):
        engine = MMDeepSpeedEngine(args=args,
                                 model=model,
                                 optimizer=optimizer,
                                 model_parameters=model_parameters,
                                 training_data=training_data,
                                 lr_scheduler=lr_scheduler,
                                 mpu=mpu,
                                 dist_init_required=dist_init_required,
                                 collate_fn=collate_fn,
                                 config=config,
                                 config_params=config_params)
    else:
        assert mpu is None, "mpu must be None with pipeline parallelism"
        engine = PipelineEngine(args=args,
                                model=model,
                                optimizer=optimizer,
                                model_parameters=model_parameters,
                                training_data=training_data,
                                lr_scheduler=lr_scheduler,
                                mpu=model.mpu(),
                                dist_init_required=dist_init_required,
                                collate_fn=collate_fn,
                                config=config,
                                config_params=config_params)

    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler
    ]
    return tuple(return_items)






