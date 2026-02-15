"""
Safety Compliance Module for Hydrogen-Electric UAV Propulsion Simulator

This module implements safety compliance checks based on EASA (European Union Aviation
Safety Agency) standards for hydrogen storage systems in aerospace applications.

Author: Senior Aerospace Propulsion Engineer
"""

import numpy as np


class SafetyCompliance:
    """
    Safety compliance checker for hydrogen storage systems.
    
    Implements EASA standards for:
    - Pressure vessel burst pressure calculations
    - Factor of Safety (FoS) requirements
    - Energy buffer (reserve fuel) calculations
    - Emergency landing fuel requirements
    """
    
    def __init__(self, tank_radius=0.3, tank_length=1.5, wall_thickness=0.002,
                 material_yield_strength=350e6, operating_pressure=500e3):
        """
        Initialize safety compliance checker.
        
        Args:
            tank_radius (float): Tank radius in meters (m). Default is 0.3 m.
            tank_length (float): Tank length in meters (m). Default is 1.5 m.
            wall_thickness (float): Tank wall thickness in meters (m). 
                                  Default is 0.002 m (2 mm).
            material_yield_strength (float): Material yield strength in Pascals (Pa).
                                           Default is 350 MPa (typical for aerospace aluminum).
            operating_pressure (float): Operating pressure in Pascals (Pa).
                                      Default is 500 kPa (0.5 MPa).
        """
        self.tank_radius = tank_radius  # m
        self.tank_length = tank_length  # m
        self.wall_thickness = wall_thickness  # m
        self.material_yield_strength = material_yield_strength  # Pa
        self.operating_pressure = operating_pressure  # Pa
        
        # EASA requirements
        self.easa_min_fos = 2.2  # Minimum Factor of Safety per EASA standards
        
        # Physical constants
        self.g = 9.81  # m/s²
    
    def calculate_burst_pressure_thin_wall(self):
        """
        Calculate burst pressure using Thin-Walled Pressure Vessel formula.
        
        For a thin-walled cylindrical pressure vessel, the hoop stress is:
        σ = PD / (2t)
        
        where:
        - σ: Hoop stress (Pa)
        - P: Internal pressure (Pa)
        - D: Diameter (m) = 2R
        - t: Wall thickness (m)
        
        Solving for burst pressure (when σ = yield strength):
        P_burst = (2t × σ_yield) / D
        
        This is the maximum pressure the tank can withstand before failure.
        
        Returns:
            float: Burst pressure in Pascals (Pa)
        """
        # Tank diameter
        diameter = 2 * self.tank_radius
        
        # Thin-walled pressure vessel formula: σ = PD/(2t)
        # Solving for P when σ = σ_yield: P_burst = (2t × σ_yield) / D
        burst_pressure = (2 * self.wall_thickness * self.material_yield_strength) / diameter
        
        return burst_pressure
    
    def calculate_factor_of_safety(self):
        """
        Calculate Factor of Safety (FoS) for the pressure vessel.
        
        Factor of Safety = Burst Pressure / Operating Pressure
        
        EASA requires minimum FoS of 2.2 for hydrogen storage systems in aircraft.
        This ensures adequate safety margin for:
        - Material property variations
        - Manufacturing tolerances
        - Fatigue and aging effects
        - Unexpected operating conditions
        
        Returns:
            float: Factor of Safety (dimensionless)
        """
        burst_pressure = self.calculate_burst_pressure_thin_wall()
        fos = burst_pressure / self.operating_pressure if self.operating_pressure > 0 else np.inf
        
        return fos
    
    def check_easa_compliance(self):
        """
        Check compliance with EASA standards for hydrogen storage.
        
        EASA Requirements:
        - Minimum Factor of Safety: 2.2
        - Pressure vessel must be designed for maximum operating pressure
        - Adequate material properties and manufacturing quality
        
        Returns:
            dict: Dictionary containing compliance status and details
        """
        fos = self.calculate_factor_of_safety()
        burst_pressure = self.calculate_burst_pressure_thin_wall()
        
        is_compliant = fos >= self.easa_min_fos
        
        return {
            'is_compliant': is_compliant,
            'factor_of_safety': fos,
            'burst_pressure_Pa': burst_pressure,
            'burst_pressure_kPa': burst_pressure / 1000,
            'operating_pressure_Pa': self.operating_pressure,
            'operating_pressure_kPa': self.operating_pressure / 1000,
            'easa_min_fos': self.easa_min_fos,
            'margin': fos - self.easa_min_fos,
            'compliance_status': 'COMPLIANT' if is_compliant else 'NON-COMPLIANT'
        }
    
    def calculate_energy_buffer(self, cruise_range_km, reserve_time_minutes=30,
                               cruise_velocity=30.0, lift_to_drag=15.0,
                               total_efficiency=0.5):
        """
        Calculate Energy Buffer (Reserve Fuel) for emergency landing scenarios.
        
        The Energy Buffer represents the minimum fuel that must remain in the tank
        to safely reach an alternate landing site if the destination airport is closed.
        
        This follows EASA requirements for reserve fuel:
        - Reserve fuel for alternate airport diversion
        - Reserve fuel for holding pattern (typically 30 minutes)
        - Reserve fuel for final approach and landing
        
        Args:
            cruise_range_km (float): Normal cruise range in kilometers
            reserve_time_minutes (float): Reserve flight time in minutes.
                                         Default is 30 minutes (EASA standard).
            cruise_velocity (float): Cruise velocity in m/s. Default is 30.0 m/s.
            lift_to_drag (float): Lift-to-drag ratio. Default is 15.0.
            total_efficiency (float): Total system efficiency. Default is 0.5.
        
        Returns:
            dict: Dictionary containing energy buffer calculations
        """
        # Convert reserve time to seconds
        reserve_time_seconds = reserve_time_minutes * 60
        
        # Calculate reserve distance
        reserve_distance_m = cruise_velocity * reserve_time_seconds
        reserve_distance_km = reserve_distance_m / 1000
        
        # Calculate energy required for reserve flight
        # Using modified Breguet equation in reverse
        hydrogen_energy_density = 120e6  # J/kg
        
        # Estimate average mass during reserve flight (assume 80% of cruise mass)
        # This is a simplified assumption - in practice, would use actual mission profile
        avg_mass_ratio = 0.8
        
        # Energy required for reserve flight
        # E = (m × g × R) / (η × L/D)
        # Solving for fuel mass: m_fuel = E / e_H2
        # Simplified: m_fuel ≈ (R × g × m_avg) / (η × L/D × e_H2)
        
        # For more accurate calculation, use Breguet equation
        # R = (η × e_H2 / g) × (L/D) × ln(m_start / m_end)
        # Solving for fuel mass needed for reserve range
        
        # Simplified calculation: assume linear relationship for small fuel consumption
        # Reserve fuel ≈ (reserve_distance / cruise_range) × total_fuel_consumed
        
        # More accurate: use Breguet equation
        # For reserve flight, estimate start and end masses
        # Assume reserve flight uses 10-20% of total fuel
        
        # Calculate reserve fuel mass using energy balance
        # Power required = (m × g × v) / (L/D × η)
        # Energy = Power × time
        # Fuel mass = Energy / (e_H2 × η)
        
        # Average aircraft mass during reserve (simplified)
        # Assume payload + empty mass + 50% of fuel
        avg_aircraft_mass = 15.0  # kg (simplified estimate)
        
        # Power required for level flight
        power_required = (avg_aircraft_mass * self.g * cruise_velocity) / \
                        (lift_to_drag * total_efficiency)
        
        # Energy required for reserve flight
        energy_required = power_required * reserve_time_seconds
        
        # Fuel mass required (accounting for efficiency)
        reserve_fuel_mass = energy_required / (hydrogen_energy_density * total_efficiency)
        
        # Alternative calculation using Breguet equation
        # More accurate for longer reserves
        start_mass_reserve = avg_aircraft_mass + reserve_fuel_mass
        end_mass_reserve = avg_aircraft_mass
        
        # Calculate reserve range using Breguet
        reserve_range_breguet = (total_efficiency * hydrogen_energy_density / self.g) * \
                               lift_to_drag * np.log(start_mass_reserve / end_mass_reserve)
        
        # Iterate to find correct reserve fuel mass
        # Use iterative approach for accuracy
        reserve_fuel_accurate = reserve_fuel_mass
        for _ in range(5):  # Few iterations for convergence
            start_mass = avg_aircraft_mass + reserve_fuel_accurate
            end_mass = avg_aircraft_mass
            calculated_range = (total_efficiency * hydrogen_energy_density / self.g) * \
                             lift_to_drag * np.log(start_mass / end_mass)
            
            # Adjust fuel mass based on range difference
            if calculated_range > 0:
                ratio = reserve_distance_m / calculated_range
                reserve_fuel_accurate = reserve_fuel_accurate * ratio
        
        reserve_fuel_mass = max(0.01, reserve_fuel_accurate)  # Minimum 10g
        
        return {
            'reserve_fuel_mass_kg': reserve_fuel_mass,
            'reserve_distance_km': reserve_distance_km,
            'reserve_time_minutes': reserve_time_minutes,
            'energy_required_J': energy_required,
            'power_required_W': power_required,
            'recommendation': f'Maintain minimum {reserve_fuel_mass*1000:.1f} g of fuel for emergency landing'
        }
    
    def calculate_minimum_safe_fuel(self, total_fuel_mass, cruise_range_km,
                                   reserve_time_minutes=30, cruise_velocity=30.0,
                                   lift_to_drag=15.0, total_efficiency=0.5):
        """
        Calculate minimum safe fuel level considering reserve requirements.
        
        This determines the fuel level at which the aircraft must divert to
        alternate airport or land immediately to maintain safety margins.
        
        Args:
            total_fuel_mass (float): Total fuel mass in kg
            cruise_range_km (float): Normal cruise range in km
            reserve_time_minutes (float): Reserve time in minutes. Default is 30.
            cruise_velocity (float): Cruise velocity in m/s. Default is 30.0.
            lift_to_drag (float): Lift-to-drag ratio. Default is 15.0.
            total_efficiency (float): Total system efficiency. Default is 0.5.
        
        Returns:
            dict: Dictionary containing minimum safe fuel calculations
        """
        energy_buffer = self.calculate_energy_buffer(
            cruise_range_km, reserve_time_minutes, cruise_velocity,
            lift_to_drag, total_efficiency
        )
        
        reserve_fuel = energy_buffer['reserve_fuel_mass_kg']
        minimum_safe_fuel = reserve_fuel
        
        # Add 10% safety margin
        safety_margin = 0.1
        minimum_safe_fuel_with_margin = reserve_fuel * (1 + safety_margin)
        
        # Calculate usable fuel (total - reserve)
        usable_fuel = total_fuel_mass - minimum_safe_fuel_with_margin
        
        # Calculate range with usable fuel
        usable_range_km = cruise_range_km * (usable_fuel / total_fuel_mass) if total_fuel_mass > 0 else 0
        
        return {
            'total_fuel_mass_kg': total_fuel_mass,
            'reserve_fuel_mass_kg': reserve_fuel,
            'minimum_safe_fuel_kg': minimum_safe_fuel_with_margin,
            'usable_fuel_mass_kg': max(0, usable_fuel),
            'usable_range_km': max(0, usable_range_km),
            'reserve_percentage': (reserve_fuel / total_fuel_mass * 100) if total_fuel_mass > 0 else 0,
            'warning_threshold_kg': minimum_safe_fuel_with_margin,
            'critical_threshold_kg': reserve_fuel
        }
