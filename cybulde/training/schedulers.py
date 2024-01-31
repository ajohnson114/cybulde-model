from abc import ABC, abstractmethod
from typing import Any, Protocol, Optional, Union, Literal

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

class PartialSchedulerType(Protocol):
    def __call__(
        self, optimizer: Optimizer, estimated_stepping_batches: Optional[Union[int,float]] = None
        ) -> _LRScheduler:
        """
        optimizer is the optimizer
        estimated_stepping_batches is used when you use a scheduler that depends on the current epoch
        It is the estimated number of batches that will optimizer.step() during training
        """
        ...

class LightningScheduler(ABC):
    def __init__(
        self,
        scheduler: PartialSchedulerType,
        interval: Literal['epoch','step'] = 'epoch',
        frequency: int = 1,
        monitor: str = "val_loss",
        strict: bool = True,
        name: Optional[str] = None
    )->None:
        """
        See here for more info: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        
        Important bits here:

        # REQUIRED: The scheduler instance
        scheduler: lr_scheduler,

        # The unit of the scheduler's step size, could also be 'step'.
        # 'epoch' updates the scheduler on epoch end whereas 'step'
        # updates it after a optimizer update.
        interval: "epoch",

        # How many epochs/steps should pass between calls to
        # `scheduler.step()`. 1 corresponds to updating the learning
        # rate after every epoch/step.
        frequency: 1,

        # Metric to monitor for schedulers like `ReduceLROnPlateau`
        "monitor": "val_loss",

        # If set to `True`, will enforce that the value specified 'monitor'
        # is available when the scheduler is updated, thus stopping
        # training if not found. If set to `False`, it will only produce a warning
        "strict": True,

        # If using the `LearningRateMonitor` callback to monitor the
        # learning rate progress, this keyword can be used to specify
        # a custom logged name
        "name": None,
        """
        self.scheduler = scheduler
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor
        self.strict = strict
        self.name = name

    
    @abstractmethod
    def configure_scheduler(self, optimizer: Optimizer, estimated_stepping_batches: Union[int,float]) -> dict[str, Any]:
        ...

class CommonLightningScheduler(LightningScheduler):
    def configure_scheduler(self, optimizer: Optimizer, estimated_stepping_batches: Union[int,float]) -> dict[str, Any]:
        return {
            'scheduler': self.scheduler(optimizer),
            "interval": self.interval,
            "frequency": self.frequency,
            "monitor": self.monitor,
            "strict": self.strict,
            "name": self.name
        }