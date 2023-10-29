from dataclasses import dataclass
from typing import List

from gp_detection.models.parameter_descriptor import ParameterDescriptor


@dataclass
class GepsilonModelDescriptor:
    name: str
    parameters: List[ParameterDescriptor]
