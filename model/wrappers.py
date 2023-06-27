import torch
from typing import Any
from typing import Union
from typing import Callable
from torchmetrics import Metric
from torch.optim.optimizer import Optimizer


class WrappedOptimizer(object):
    def __init__(self, optimizer: Union[Optimizer, Any], **opt_params):
        self.opt = optimizer
        self.params = opt_params

    def __call__(self, model_params):
        # noinspection PyCallingNonCallable
        return self.opt(params=model_params, **self.params)


class WrappedLoss(object):
    def __init__(self, loss_fn: Callable, **extra_params):
        # noinspection SpellCheckingInspection
        """
        Wrap a loss function
        Args:
            loss_fn: target function,
                must take preds and targets as keyword arguments
            **extra_params: additional keyword arguments for loss_fn
        """
        self.loss_fn = loss_fn
        self.extra_params = extra_params

    # noinspection SpellCheckingInspection
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor):
        return self.loss_fn(preds=preds, targets=targets, **self.extra_params)


class WrappedScheduler(object):
    def __init__(self, scheduler: Callable, **extra_params):
        """
        Wrap a scheduler
        Args:
            scheduler: target scheduler class
            **extra_params: additional keyword arguments for the scheduler
        """
        self.scheduler = scheduler
        self.extra_params = extra_params

    def __call__(self, optimizer):
        return self.scheduler(optimizer=optimizer, **self.extra_params)


class WrappedMetric(object):
    def __init__(self, metric: Metric, **kwargs: Any):
        self.metric = metric
        self.kwargs = kwargs

    def get_metric(self, **extra_args):
        kwargs = self.kwargs.copy()
        kwargs.update(extra_args)
