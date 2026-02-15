"""
Hydrogen-Electric UAV Propulsion Simulator - Core Module

This package contains the fundamental physics and engineering models for
hydrogen-electric propulsion systems in aerospace applications.
"""

try:
    from .physics import HydrogenProperties
    HAS_PHYSICS = True
except ImportError:
    HAS_PHYSICS = False
    HydrogenProperties = None

from .fuel_cell import PEMFuelCell
from .tank import LH2Tank
from .mission import MissionProfile, monte_carlo_mission_analysis, stochastic_mission_analysis
from .propulsion import PropulsionSystem
from .safety_compliance import SafetyCompliance
from .assistant import PropulsionAssistant

if HAS_PHYSICS:
    __all__ = ['HydrogenProperties', 'PEMFuelCell', 'LH2Tank', 'MissionProfile', 
               'PropulsionSystem', 'SafetyCompliance', 'PropulsionAssistant']
else:
    __all__ = ['PEMFuelCell', 'LH2Tank', 'MissionProfile', 'PropulsionSystem', 
               'SafetyCompliance', 'PropulsionAssistant']
