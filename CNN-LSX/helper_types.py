
from dataclasses import dataclass
from typing import Any, Optional

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Loaders:
    train: DataLoader[Any]
    critic: Optional[DataLoader[Any]]
    test: Optional[DataLoader[Any]]
    visualization: Optional[DataLoader[Any]]


@dataclass
class Logging:
    """Combines the variables that are only used for logging"""
    writer: SummaryWriter
    run_name: str
    log_interval: int
    log_interval_accuracy: int
    critic_log_interval: int


