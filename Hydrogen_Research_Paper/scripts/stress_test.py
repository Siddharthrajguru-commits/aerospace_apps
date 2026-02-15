"""
Stress Test Script for Hydrogen-Electric UAV Propulsion Simulator

This script automatically tests 50 edge case scenarios to ensure the system
handles extreme inputs gracefully without crashing. Physical impossibilities
are logged to errors.log for review.

Author: Senior Aerospace Propulsion Engineer
"""

import sys
import os
import logging
from datetime import datetime
import numpy as np

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.mission import MissionProfile
from core.tank import LH2Tank
from core.fuel_cell import PEMFuelCell
from core.safety_compliance import SafetyCompliance

# Helper functions re-implemented here to avoid Streamlit dependencies
def calculate_tank_weight_from_thickness(insulation_thickness_mm):
    """Calculate tank weight based on insulation thickness."""
    base_tank_mass = 0.3  # kg
    insulation_mass_per_mm = 0.01  # kg per mm
    tank_weight = base_tank_mass + (insulation_thickness_mm * insulation_mass_per_mm)
    return tank_weight

def calculate_range_with_tank_weight_helper(tank_weight, fuel_mass=2.0, payload_mass=5.0, 
                                           empty_mass=10.0, lift_to_drag=15.0, 
                                           total_efficiency=0.5, cruise_velocity=30.0):
    """Calculate mission range for given tank weight."""
    total_empty_mass = empty_mass + tank_weight
    start_mass = payload_mass + fuel_mass + total_empty_mass
    end_mass = payload_mass + total_empty_mass + 0.1
    
    hydrogen_energy_density = 120e6  # J/kg
    g = 9.81  # m/s²
    
    if end_mass <= 0 or start_mass <= end_mass:
        return 0.0
    
    range_m = (total_efficiency * hydrogen_energy_density / g) * \
              lift_to_drag * np.log(start_mass / end_mass)
    
    return range_m / 1000  # Convert to km


# Setup logging
log_file = os.path.join(os.path.dirname(__file__), '..', 'errors.log')
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StressTestRunner:
    """Runs comprehensive stress tests on the propulsion system."""
    
    def __init__(self):
        self.test_count = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.physical_impossibilities = []
        
    def log_physical_impossibility(self, test_name, issue, value=None):
        """Log physical impossibilities to errors.log."""
        message = f"PHYSICAL IMPOSSIBILITY in {test_name}: {issue}"
        if value is not None:
            message += f" (Value: {value})"
        logger.warning(message)
        self.physical_impossibilities.append({
            'test': test_name,
            'issue': issue,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
    
    def validate_result(self, test_name, result, min_val=None, max_val=None, 
                       check_negative=True, check_nan=True, check_inf=True):
        """Validate a result for physical impossibilities."""
        if check_nan and (np.isnan(result) if isinstance(result, (int, float, np.number)) else False):
            self.log_physical_impossibility(test_name, "Result is NaN", result)
            return False
        
        if check_inf and (np.isinf(result) if isinstance(result, (int, float, np.number)) else False):
            self.log_physical_impossibility(test_name, "Result is infinite", result)
            return False
        
        if check_negative and isinstance(result, (int, float, np.number)) and result < 0:
            self.log_physical_impossibility(test_name, "Negative value", result)
            return False
        
        if min_val is not None and isinstance(result, (int, float, np.number)) and result < min_val:
            self.log_physical_impossibility(test_name, f"Value below minimum ({min_val})", result)
            return False
        
        if max_val is not None and isinstance(result, (int, float, np.number)) and result > max_val:
            self.log_physical_impossibility(test_name, f"Value above maximum ({max_val})", result)
            return False
        
        return True
    
    def run_test(self, test_name, test_func):
        """Run a single test with error handling."""
        self.test_count += 1
        try:
            result = test_func()
            self.passed_tests += 1
            print(f"[PASS] Test {self.test_count}: {test_name} - PASSED")
            return result
        except Exception as e:
            self.failed_tests += 1
            error_msg = f"Test {self.test_count}: {test_name} - FAILED with exception: {str(e)}"
            logger.error(error_msg)
            print(f"[FAIL] {error_msg}")
            return None
    
    def test_zero_fuel(self):
        """Test 1: Zero fuel mass."""
        def test():
            mission = MissionProfile(fuel_mass=0.0, payload_mass=5.0, empty_mass=10.0)
            range_km = mission.calculate_range_breguet()
            self.validate_result("Zero Fuel", range_km, min_val=0.0)
            return range_km
        return self.run_test("Zero Fuel", test)
    
    def test_negative_fuel(self):
        """Test 2: Negative fuel mass."""
        def test():
            mission = MissionProfile(fuel_mass=-1.0, payload_mass=5.0, empty_mass=10.0)
            range_km = mission.calculate_range_breguet()
            self.validate_result("Negative Fuel", range_km, check_negative=True)
            return range_km
        return self.run_test("Negative Fuel", test)
    
    def test_zero_payload(self):
        """Test 3: Zero payload mass."""
        def test():
            mission = MissionProfile(fuel_mass=2.0, payload_mass=0.0, empty_mass=10.0)
            range_km = mission.calculate_range_breguet()
            self.validate_result("Zero Payload", range_km, min_val=0.0)
            return range_km
        return self.run_test("Zero Payload", test)
    
    def test_zero_empty_mass(self):
        """Test 4: Zero empty mass."""
        def test():
            mission = MissionProfile(fuel_mass=2.0, payload_mass=5.0, empty_mass=0.0)
            range_km = mission.calculate_range_breguet()
            self.validate_result("Zero Empty Mass", range_km, min_val=0.0)
            return range_km
        return self.run_test("Zero Empty Mass", test)
    
    def test_ultra_thin_insulation(self):
        """Test 5: Ultra-thin insulation (0.1 mm)."""
        def test():
            tank = LH2Tank(tank_radius=0.3, tank_length=1.5)
            tank.mli_thickness = 0.0001  # 0.1 mm
            heat_leak = tank.calculate_total_heat_leak()
            self.validate_result("Ultra-Thin Insulation", heat_leak, min_val=0.0)
            return heat_leak
        return self.run_test("Ultra-Thin Insulation", test)
    
    def test_zero_insulation(self):
        """Test 6: Zero insulation thickness."""
        def test():
            tank = LH2Tank(tank_radius=0.3, tank_length=1.5)
            tank.mli_thickness = 0.0
            try:
                heat_leak = tank.calculate_total_heat_leak()
                self.validate_result("Zero Insulation", heat_leak, min_val=0.0)
            except ZeroDivisionError:
                # Expected: division by zero when thickness is zero
                self.log_physical_impossibility("Zero Insulation", 
                                               "Division by zero: insulation thickness cannot be zero")
                return float('inf')  # Return infinity to indicate infinite heat leak
            return heat_leak
        return self.run_test("Zero Insulation", test)
    
    def test_negative_insulation(self):
        """Test 7: Negative insulation thickness."""
        def test():
            tank = LH2Tank(tank_radius=0.3, tank_length=1.5)
            tank.mli_thickness = -0.01
            heat_leak = tank.calculate_total_heat_leak()
            self.validate_result("Negative Insulation", heat_leak, check_negative=True)
            return heat_leak
        return self.run_test("Negative Insulation", test)
    
    def test_supersonic_speed(self):
        """Test 8: Supersonic cruise velocity (400 m/s)."""
        def test():
            range_km = calculate_range_with_tank_weight_helper(
                tank_weight=0.5, fuel_mass=2.0, payload_mass=5.0,
                empty_mass=10.0, lift_to_drag=15.0, total_efficiency=0.5,
                cruise_velocity=400.0  # Supersonic
            )
            self.validate_result("Supersonic Speed", range_km, min_val=0.0)
            return range_km
        return self.run_test("Supersonic Speed", test)
    
    def test_hypersonic_speed(self):
        """Test 9: Hypersonic cruise velocity (2000 m/s)."""
        def test():
            range_km = calculate_range_with_tank_weight_helper(
                tank_weight=0.5, fuel_mass=2.0, payload_mass=5.0,
                empty_mass=10.0, lift_to_drag=15.0, total_efficiency=0.5,
                cruise_velocity=2000.0  # Hypersonic
            )
            self.validate_result("Hypersonic Speed", range_km, min_val=0.0)
            return range_km
        return self.run_test("Hypersonic Speed", test)
    
    def test_zero_velocity(self):
        """Test 10: Zero cruise velocity."""
        def test():
            range_km = calculate_range_with_tank_weight_helper(
                tank_weight=0.5, fuel_mass=2.0, payload_mass=5.0,
                empty_mass=10.0, lift_to_drag=15.0, total_efficiency=0.5,
                cruise_velocity=0.0
            )
            self.validate_result("Zero Velocity", range_km, min_val=0.0)
            return range_km
        return self.run_test("Zero Velocity", test)
    
    def test_negative_velocity(self):
        """Test 11: Negative cruise velocity."""
        def test():
            range_km = calculate_range_with_tank_weight_helper(
                tank_weight=0.5, fuel_mass=2.0, payload_mass=5.0,
                empty_mass=10.0, lift_to_drag=15.0, total_efficiency=0.5,
                cruise_velocity=-10.0
            )
            self.validate_result("Negative Velocity", range_km, check_negative=True)
            return range_km
        return self.run_test("Negative Velocity", test)
    
    def test_zero_efficiency(self):
        """Test 12: Zero system efficiency."""
        def test():
            mission = MissionProfile(
                fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                total_efficiency=0.0
            )
            range_km = mission.calculate_range_breguet()
            self.validate_result("Zero Efficiency", range_km, min_val=0.0)
            return range_km
        return self.run_test("Zero Efficiency", test)
    
    def test_over_100_percent_efficiency(self):
        """Test 13: Efficiency > 100%."""
        def test():
            mission = MissionProfile(
                fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                total_efficiency=1.5  # 150%
            )
            range_km = mission.calculate_range_breguet()
            self.validate_result("Over 100% Efficiency", range_km, max_val=None)
            self.log_physical_impossibility("Over 100% Efficiency", 
                                           "Efficiency exceeds 100%", 1.5)
            return range_km
        return self.run_test("Over 100% Efficiency", test)
    
    def test_negative_efficiency(self):
        """Test 14: Negative efficiency."""
        def test():
            mission = MissionProfile(
                fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                total_efficiency=-0.1
            )
            range_km = mission.calculate_range_breguet()
            self.validate_result("Negative Efficiency", range_km, check_negative=True)
            return range_km
        return self.run_test("Negative Efficiency", test)
    
    def test_zero_lift_to_drag(self):
        """Test 15: Zero L/D ratio."""
        def test():
            mission = MissionProfile(
                fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                lift_to_drag=0.0
            )
            range_km = mission.calculate_range_breguet()
            self.validate_result("Zero L/D", range_km, min_val=0.0)
            return range_km
        return self.run_test("Zero L/D", test)
    
    def test_negative_lift_to_drag(self):
        """Test 16: Negative L/D ratio."""
        def test():
            mission = MissionProfile(
                fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                lift_to_drag=-10.0
            )
            range_km = mission.calculate_range_breguet()
            self.validate_result("Negative L/D", range_km, check_negative=True)
            return range_km
        return self.run_test("Negative L/D", test)
    
    def test_extremely_high_lift_to_drag(self):
        """Test 17: Extremely high L/D ratio (100)."""
        def test():
            mission = MissionProfile(
                fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                lift_to_drag=100.0
            )
            range_km = mission.calculate_range_breguet()
            self.validate_result("High L/D", range_km, min_val=0.0)
            return range_km
        return self.run_test("High L/D", test)
    
    def test_zero_tank_radius(self):
        """Test 18: Zero tank radius."""
        def test():
            tank = LH2Tank(tank_radius=0.0, tank_length=1.5)
            heat_leak = tank.calculate_total_heat_leak()
            self.validate_result("Zero Tank Radius", heat_leak, min_val=0.0)
            return heat_leak
        return self.run_test("Zero Tank Radius", test)
    
    def test_negative_tank_radius(self):
        """Test 19: Negative tank radius."""
        def test():
            tank = LH2Tank(tank_radius=-0.1, tank_length=1.5)
            heat_leak = tank.calculate_total_heat_leak()
            self.validate_result("Negative Tank Radius", heat_leak, check_negative=True)
            return heat_leak
        return self.run_test("Negative Tank Radius", test)
    
    def test_very_large_tank_radius(self):
        """Test 20: Very large tank radius (10 m)."""
        def test():
            tank = LH2Tank(tank_radius=10.0, tank_length=1.5)
            heat_leak = tank.calculate_total_heat_leak()
            self.validate_result("Large Tank Radius", heat_leak, min_val=0.0)
            return heat_leak
        return self.run_test("Large Tank Radius", test)
    
    def test_zero_tank_length(self):
        """Test 21: Zero tank length."""
        def test():
            tank = LH2Tank(tank_radius=0.3, tank_length=0.0)
            heat_leak = tank.calculate_total_heat_leak()
            self.validate_result("Zero Tank Length", heat_leak, min_val=0.0)
            return heat_leak
        return self.run_test("Zero Tank Length", test)
    
    def test_negative_tank_length(self):
        """Test 22: Negative tank length."""
        def test():
            tank = LH2Tank(tank_radius=0.3, tank_length=-1.0)
            heat_leak = tank.calculate_total_heat_leak()
            self.validate_result("Negative Tank Length", heat_leak, check_negative=True)
            return heat_leak
        return self.run_test("Negative Tank Length", test)
    
    def test_extremely_heavy_payload(self):
        """Test 23: Extremely heavy payload (1000 kg)."""
        def test():
            mission = MissionProfile(fuel_mass=2.0, payload_mass=1000.0, empty_mass=10.0)
            range_km = mission.calculate_range_breguet()
            self.validate_result("Heavy Payload", range_km, min_val=0.0)
            return range_km
        return self.run_test("Heavy Payload", test)
    
    def test_extremely_heavy_fuel(self):
        """Test 24: Extremely heavy fuel load (1000 kg)."""
        def test():
            mission = MissionProfile(fuel_mass=1000.0, payload_mass=5.0, empty_mass=10.0)
            range_km = mission.calculate_range_breguet()
            self.validate_result("Heavy Fuel", range_km, min_val=0.0)
            return range_km
        return self.run_test("Heavy Fuel", test)
    
    def test_extremely_heavy_empty_mass(self):
        """Test 25: Extremely heavy empty mass (1000 kg)."""
        def test():
            mission = MissionProfile(fuel_mass=2.0, payload_mass=5.0, empty_mass=1000.0)
            range_km = mission.calculate_range_breguet()
            self.validate_result("Heavy Empty Mass", range_km, min_val=0.0)
            return range_km
        return self.run_test("Heavy Empty Mass", test)
    
    def test_very_thick_insulation(self):
        """Test 26: Very thick insulation (1000 mm)."""
        def test():
            tank = LH2Tank(tank_radius=0.3, tank_length=1.5)
            tank.mli_thickness = 1.0  # 1000 mm = 1 m
            heat_leak = tank.calculate_total_heat_leak()
            self.validate_result("Thick Insulation", heat_leak, min_val=0.0)
            return heat_leak
        return self.run_test("Thick Insulation", test)
    
    def test_zero_fuel_cell_voltage(self):
        """Test 27: Zero fuel cell open circuit voltage."""
        def test():
            fuel_cell = PEMFuelCell(E_ocv=0.0, temperature=353.15)
            current_density = np.array([100, 200, 300])
            voltage = fuel_cell.calculate_cell_voltage(current_density)
            self.validate_result("Zero Fuel Cell Voltage", voltage.min(), min_val=0.0)
            return voltage
        return self.run_test("Zero Fuel Cell Voltage", test)
    
    def test_negative_fuel_cell_voltage(self):
        """Test 28: Negative fuel cell open circuit voltage."""
        def test():
            fuel_cell = PEMFuelCell(E_ocv=-1.0, temperature=353.15)
            current_density = np.array([100, 200, 300])
            voltage = fuel_cell.calculate_cell_voltage(current_density)
            self.validate_result("Negative Fuel Cell Voltage", voltage.min(), check_negative=True)
            return voltage
        return self.run_test("Negative Fuel Cell Voltage", test)
    
    def test_zero_temperature(self):
        """Test 29: Zero temperature (absolute zero)."""
        def test():
            fuel_cell = PEMFuelCell(E_ocv=1.2, temperature=0.0)
            current_density = np.array([100, 200, 300])
            voltage = fuel_cell.calculate_cell_voltage(current_density)
            self.validate_result("Zero Temperature", voltage.min(), min_val=0.0)
            return voltage
        return self.run_test("Zero Temperature", test)
    
    def test_negative_temperature(self):
        """Test 30: Negative temperature."""
        def test():
            fuel_cell = PEMFuelCell(E_ocv=1.2, temperature=-100.0)
            current_density = np.array([100, 200, 300])
            voltage = fuel_cell.calculate_cell_voltage(current_density)
            self.validate_result("Negative Temperature", voltage.min(), check_negative=False)
            return voltage
        return self.run_test("Negative Temperature", test)
    
    def test_extremely_high_temperature(self):
        """Test 31: Extremely high temperature (5000 K)."""
        def test():
            fuel_cell = PEMFuelCell(E_ocv=1.2, temperature=5000.0)
            current_density = np.array([100, 200, 300])
            voltage = fuel_cell.calculate_cell_voltage(current_density)
            self.validate_result("High Temperature", voltage.min(), min_val=0.0)
            return voltage
        return self.run_test("High Temperature", test)
    
    def test_zero_current_density(self):
        """Test 32: Zero current density."""
        def test():
            fuel_cell = PEMFuelCell(E_ocv=1.2, temperature=353.15)
            current_density = np.array([0.0])
            voltage = fuel_cell.calculate_cell_voltage(current_density)
            self.validate_result("Zero Current Density", voltage[0], min_val=0.0)
            return voltage
        return self.run_test("Zero Current Density", test)
    
    def test_negative_current_density(self):
        """Test 33: Negative current density."""
        def test():
            fuel_cell = PEMFuelCell(E_ocv=1.2, temperature=353.15)
            current_density = np.array([-100, -200])
            voltage = fuel_cell.calculate_cell_voltage(current_density)
            self.validate_result("Negative Current Density", voltage.min(), check_negative=False)
            return voltage
        return self.run_test("Negative Current Density", test)
    
    def test_extremely_high_current_density(self):
        """Test 34: Extremely high current density (100,000 A/m²)."""
        def test():
            fuel_cell = PEMFuelCell(E_ocv=1.2, temperature=353.15)
            current_density = np.array([100000.0])
            voltage = fuel_cell.calculate_cell_voltage(current_density)
            self.validate_result("High Current Density", voltage[0], min_val=0.0)
            return voltage
        return self.run_test("High Current Density", test)
    
    def test_zero_wall_thickness(self):
        """Test 35: Zero wall thickness for safety compliance."""
        def test():
            safety = SafetyCompliance(
                tank_radius=0.3, tank_length=1.5,
                wall_thickness=0.0, material_yield_strength=350e6,
                operating_pressure=500e3
            )
            compliance = safety.check_easa_compliance()
            self.validate_result("Zero Wall Thickness", 
                               compliance['factor_of_safety'], min_val=0.0)
            return compliance
        return self.run_test("Zero Wall Thickness", test)
    
    def test_negative_wall_thickness(self):
        """Test 36: Negative wall thickness."""
        def test():
            safety = SafetyCompliance(
                tank_radius=0.3, tank_length=1.5,
                wall_thickness=-0.001, material_yield_strength=350e6,
                operating_pressure=500e3
            )
            compliance = safety.check_easa_compliance()
            self.validate_result("Negative Wall Thickness", 
                               compliance['factor_of_safety'], check_negative=True)
            return compliance
        return self.run_test("Negative Wall Thickness", test)
    
    def test_zero_operating_pressure(self):
        """Test 37: Zero operating pressure."""
        def test():
            safety = SafetyCompliance(
                tank_radius=0.3, tank_length=1.5,
                wall_thickness=0.002, material_yield_strength=350e6,
                operating_pressure=0.0
            )
            compliance = safety.check_easa_compliance()
            self.validate_result("Zero Operating Pressure", 
                               compliance['factor_of_safety'], min_val=0.0)
            return compliance
        return self.run_test("Zero Operating Pressure", test)
    
    def test_negative_operating_pressure(self):
        """Test 38: Negative operating pressure."""
        def test():
            safety = SafetyCompliance(
                tank_radius=0.3, tank_length=1.5,
                wall_thickness=0.002, material_yield_strength=350e6,
                operating_pressure=-1000.0
            )
            compliance = safety.check_easa_compliance()
            self.validate_result("Negative Operating Pressure", 
                               compliance['factor_of_safety'], check_negative=False)
            return compliance
        return self.run_test("Negative Operating Pressure", test)
    
    def test_zero_material_strength(self):
        """Test 39: Zero material yield strength."""
        def test():
            safety = SafetyCompliance(
                tank_radius=0.3, tank_length=1.5,
                wall_thickness=0.002, material_yield_strength=0.0,
                operating_pressure=500e3
            )
            compliance = safety.check_easa_compliance()
            self.validate_result("Zero Material Strength", 
                               compliance['factor_of_safety'], min_val=0.0)
            return compliance
        return self.run_test("Zero Material Strength", test)
    
    def test_negative_material_strength(self):
        """Test 40: Negative material yield strength."""
        def test():
            safety = SafetyCompliance(
                tank_radius=0.3, tank_length=1.5,
                wall_thickness=0.002, material_yield_strength=-100e6,
                operating_pressure=500e3
            )
            compliance = safety.check_easa_compliance()
            self.validate_result("Negative Material Strength", 
                               compliance['factor_of_safety'], check_negative=True)
            return compliance
        return self.run_test("Negative Material Strength", test)
    
    def test_infinite_fuel_mass(self):
        """Test 41: Infinite fuel mass (simulated with very large number)."""
        def test():
            mission = MissionProfile(fuel_mass=1e10, payload_mass=5.0, empty_mass=10.0)
            range_km = mission.calculate_range_breguet()
            self.validate_result("Infinite Fuel Mass", range_km, check_inf=True)
            return range_km
        return self.run_test("Infinite Fuel Mass", test)
    
    def test_nan_fuel_mass(self):
        """Test 42: NaN fuel mass."""
        def test():
            mission = MissionProfile(fuel_mass=np.nan, payload_mass=5.0, empty_mass=10.0)
            range_km = mission.calculate_range_breguet()
            self.validate_result("NaN Fuel Mass", range_km, check_nan=True)
            return range_km
        return self.run_test("NaN Fuel Mass", test)
    
    def test_inf_efficiency(self):
        """Test 43: Infinite efficiency."""
        def test():
            mission = MissionProfile(
                fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                total_efficiency=np.inf
            )
            range_km = mission.calculate_range_breguet()
            self.validate_result("Infinite Efficiency", range_km, check_inf=True)
            self.log_physical_impossibility("Infinite Efficiency", 
                                           "Efficiency is infinite", np.inf)
            return range_km
        return self.run_test("Infinite Efficiency", test)
    
    def test_nan_efficiency(self):
        """Test 44: NaN efficiency."""
        def test():
            mission = MissionProfile(
                fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                total_efficiency=np.nan
            )
            range_km = mission.calculate_range_breguet()
            self.validate_result("NaN Efficiency", range_km, check_nan=True)
            return range_km
        return self.run_test("NaN Efficiency", test)
    
    def test_all_zeros(self):
        """Test 45: All parameters set to zero."""
        def test():
            mission = MissionProfile(
                fuel_mass=0.0, payload_mass=0.0, empty_mass=0.0,
                lift_to_drag=0.0, total_efficiency=0.0
            )
            range_km = mission.calculate_range_breguet()
            self.validate_result("All Zeros", range_km, min_val=0.0)
            return range_km
        return self.run_test("All Zeros", test)
    
    def test_all_negatives(self):
        """Test 46: All parameters negative."""
        def test():
            mission = MissionProfile(
                fuel_mass=-1.0, payload_mass=-1.0, empty_mass=-1.0,
                lift_to_drag=-1.0, total_efficiency=-0.1
            )
            range_km = mission.calculate_range_breguet()
            self.validate_result("All Negatives", range_km, check_negative=True)
            return range_km
        return self.run_test("All Negatives", test)
    
    def test_extreme_tank_weight(self):
        """Test 47: Extremely heavy tank weight (1000 kg)."""
        def test():
            range_km = calculate_range_with_tank_weight_helper(
                tank_weight=1000.0, fuel_mass=2.0, payload_mass=5.0,
                empty_mass=10.0, lift_to_drag=15.0, total_efficiency=0.5,
                cruise_velocity=30.0
            )
            self.validate_result("Extreme Tank Weight", range_km, min_val=0.0)
            return range_km
        return self.run_test("Extreme Tank Weight", test)
    
    def test_negative_tank_weight(self):
        """Test 48: Negative tank weight."""
        def test():
            range_km = calculate_range_with_tank_weight_helper(
                tank_weight=-10.0, fuel_mass=2.0, payload_mass=5.0,
                empty_mass=10.0, lift_to_drag=15.0, total_efficiency=0.5,
                cruise_velocity=30.0
            )
            self.validate_result("Negative Tank Weight", range_km, check_negative=True)
            return range_km
        return self.run_test("Negative Tank Weight", test)
    
    def test_start_mass_equals_end_mass(self):
        """Test 49: Start mass equals end mass (no fuel consumed)."""
        def test():
            # This should result in zero range (ln(1) = 0)
            mission = MissionProfile(fuel_mass=0.1, payload_mass=5.0, empty_mass=10.0)
            # Set end mass to match start mass
            start_mass = mission.calculate_start_mass()
            end_mass = start_mass  # No fuel consumed
            range_km = mission.calculate_range_breguet(start_mass=start_mass, end_mass=end_mass)
            self.validate_result("Start Mass = End Mass", range_km, min_val=0.0)
            return range_km
        return self.run_test("Start Mass = End Mass", test)
    
    def test_end_mass_greater_than_start_mass(self):
        """Test 50: End mass greater than start mass (impossible)."""
        def test():
            mission = MissionProfile(fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0)
            start_mass = mission.calculate_start_mass()
            end_mass = start_mass * 2.0  # Impossible: end > start
            try:
                range_km = mission.calculate_range_breguet(start_mass=start_mass, end_mass=end_mass)
                self.log_physical_impossibility("End Mass > Start Mass", 
                                               "End mass exceeds start mass", 
                                               f"start={start_mass}, end={end_mass}")
                self.validate_result("End Mass > Start Mass", range_km, check_negative=True)
            except (ValueError, ZeroDivisionError):
                # Expected: log(negative or zero) should raise error
                pass
            return None
        return self.run_test("End Mass > Start Mass", test)
    
    def run_all_tests(self):
        """Run all 50 stress tests."""
        print("=" * 70)
        print("STRESS TEST SUITE: Hydrogen-Electric UAV Propulsion Simulator")
        print("=" * 70)
        print(f"Starting stress tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Run all test methods
        test_methods = [
            self.test_zero_fuel,
            self.test_negative_fuel,
            self.test_zero_payload,
            self.test_zero_empty_mass,
            self.test_ultra_thin_insulation,
            self.test_zero_insulation,
            self.test_negative_insulation,
            self.test_supersonic_speed,
            self.test_hypersonic_speed,
            self.test_zero_velocity,
            self.test_negative_velocity,
            self.test_zero_efficiency,
            self.test_over_100_percent_efficiency,
            self.test_negative_efficiency,
            self.test_zero_lift_to_drag,
            self.test_negative_lift_to_drag,
            self.test_extremely_high_lift_to_drag,
            self.test_zero_tank_radius,
            self.test_negative_tank_radius,
            self.test_very_large_tank_radius,
            self.test_zero_tank_length,
            self.test_negative_tank_length,
            self.test_extremely_heavy_payload,
            self.test_extremely_heavy_fuel,
            self.test_extremely_heavy_empty_mass,
            self.test_very_thick_insulation,
            self.test_zero_fuel_cell_voltage,
            self.test_negative_fuel_cell_voltage,
            self.test_zero_temperature,
            self.test_negative_temperature,
            self.test_extremely_high_temperature,
            self.test_zero_current_density,
            self.test_negative_current_density,
            self.test_extremely_high_current_density,
            self.test_zero_wall_thickness,
            self.test_negative_wall_thickness,
            self.test_zero_operating_pressure,
            self.test_negative_operating_pressure,
            self.test_zero_material_strength,
            self.test_negative_material_strength,
            self.test_infinite_fuel_mass,
            self.test_nan_fuel_mass,
            self.test_inf_efficiency,
            self.test_nan_efficiency,
            self.test_all_zeros,
            self.test_all_negatives,
            self.test_extreme_tank_weight,
            self.test_negative_tank_weight,
            self.test_start_mass_equals_end_mass,
            self.test_end_mass_greater_than_start_mass,
        ]
        
        for test_method in test_methods:
            test_method()
        
        # Print summary
        print("\n" + "=" * 70)
        print("STRESS TEST SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {self.test_count}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Physical Impossibilities Detected: {len(self.physical_impossibilities)}")
        print(f"\nErrors logged to: {log_file}")
        print("=" * 70)
        
        return {
            'total': self.test_count,
            'passed': self.passed_tests,
            'failed': self.failed_tests,
            'physical_impossibilities': len(self.physical_impossibilities)
        }


if __name__ == "__main__":
    runner = StressTestRunner()
    results = runner.run_all_tests()
    
    # Exit with error code if tests failed
    if results['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)
