"""
Simulation module for Stage 4 explainability.
"""

from .what_if_analyzer import WhatIfAnalyzer
from .sensitivity_analysis import SensitivityAnalyzer
from .scenario_generator import ScenarioGenerator

__all__ = [
    'WhatIfAnalyzer',
    'SensitivityAnalyzer', 
    'ScenarioGenerator'
]
