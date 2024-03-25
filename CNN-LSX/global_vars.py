from typing import Optional

from helper_types import Logging

global_step: int  # increased in every explainer pretraining step, and every critic step.
LOGGING: Optional[Logging]  # tools for logging
DEVICE: str
