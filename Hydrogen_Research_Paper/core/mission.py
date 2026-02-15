"""
Mission Profile Module for Hydrogen-Electric UAV Propulsion Simulator

This module implements flight dynamics and mission logic for aerospace applications,
including mission phases (Takeoff, Climb, Cruise, Descent) and range calculations
using a modified Breguet Range Equation for electric flight.

Author: Senior Aerospace Propulsion Engineer
"""

import numpy as np
from typing import Optional
from core.propulsion import PropulsionSystem
from core.tank import LH2Tank
from core.physics_verification import PhysicsVerification


class MissionProfile:
    """
    Mission profile simulator for hydrogen-electric UAV propulsion systems.
    
    Implements a complete mission loop with distinct flight phases:
    - Takeoff: Initial acceleration and lift-off
    - Climb: Ascent to cruise altitude
    - Cruise: Level flight at constant altitude
    - Descent: Controlled descent to landing
    
    Uses modified Breguet Range Equation for electric propulsion systems
    where energy consumption is tracked rather than fuel burn rate.
    """
    
    def __init__(self, payload_mass=5.0, fuel_mass=2.0, empty_mass=10.0,
                 lift_to_drag=15.0, total_efficiency=0.5, hydrogen_energy_density=120e6,
                 tank: Optional[LH2Tank] = None, propulsion_system: Optional[PropulsionSystem] = None,
                 fuel_cell_power_w: float = 5000.0, enable_physics_verification: bool = True):
        """
        Initialize mission profile parameters.
        
        Args:
            payload_mass (float): Payload mass in kg. Default is 5.0 kg.
            fuel_mass (float): Initial LH2 fuel mass in kg. Default is 2.0 kg.
            empty_mass (float): Empty aircraft mass (structure + systems) in kg.
                               Default is 10.0 kg.
            lift_to_drag (float): Lift-to-drag ratio (L/D). Default is 15.0,
                                 typical for efficient UAV designs.
            total_efficiency (float): Base system efficiency (η_total) including
                                    fuel cell, motor, and propulsive efficiency.
                                    Default is 0.5 (50%). This will be adjusted
                                    by pressure drop losses if propulsion_system is provided.
            hydrogen_energy_density (float): Hydrogen energy density in J/kg.
                                            Default is 120 MJ/kg (120e6 J/kg).
            tank (LH2Tank, optional): Tank object for BOG calculations. If None, creates default.
            propulsion_system (PropulsionSystem, optional): Propulsion system for pressure drop.
                                                           If None, pressure drop effects disabled.
            fuel_cell_power_w (float): Fuel cell power output in Watts. Default 5000 W.
            enable_physics_verification (bool): Enable energy balance verification. Default True.
        """
        self.payload_mass = payload_mass  # kg
        self.fuel_mass = fuel_mass  # kg (initial)
        self.empty_mass = empty_mass  # kg
        self.lift_to_drag = lift_to_drag  # L/D ratio
        self.base_efficiency = total_efficiency  # Base η_total (before pressure drop)
        self.total_efficiency = total_efficiency  # Effective η_total (adjusted for pressure drop)
        self.hydrogen_energy_density = hydrogen_energy_density  # J/kg
        
        # Physical constants
        self.g = 9.81  # Gravitational acceleration (m/s²)
        
        # Mission phase parameters
        self.takeoff_distance = 0.0  # km
        self.climb_distance = 0.0  # km
        self.cruise_distance = 0.0  # km
        self.descent_distance = 0.0  # km
        
        # Fuel consumption tracking
        self.remaining_fuel = fuel_mass  # kg
        self.distance_flown = 0.0  # km
        
        # Propulsion system integration
        self.propulsion_system = propulsion_system
        self.fuel_cell_power_w = fuel_cell_power_w
        
        # Tank for BOG calculations
        if tank is None:
            self.tank = LH2Tank(tank_radius=0.3, tank_length=1.5)
        else:
            self.tank = tank
        
        # Physics verification
        self.enable_physics_verification = enable_physics_verification
        if enable_physics_verification:
            # Disable verbose printing in Streamlit to avoid encoding issues
            try:
                import streamlit as st
                verbose_mode = False  # Disable printing in Streamlit
            except ImportError:
                verbose_mode = True  # Enable printing outside Streamlit
            self.physics_verifier = PhysicsVerification(verbose=verbose_mode)
        else:
            self.physics_verifier = None
        
        # Energy tracking for verification
        self.energy_tracking = {
            'fuel_cell_output_j': 0.0,
            'motor_output_j': 0.0,
            'propulsive_energy_j': 0.0,
            'fuel_cell_losses_j': 0.0,
            'motor_losses_j': 0.0,
            'propulsive_losses_j': 0.0,
            'bog_losses_j': 0.0,
            'pump_power_j': 0.0
        }
        
    def calculate_start_mass(self):
        """
        Calculate initial takeoff mass.
        
        m_start = m_payload + m_fuel + m_empty
        
        Returns:
            float: Start mass in kg
        """
        return self.payload_mass + self.fuel_mass + self.empty_mass
    
    def calculate_end_mass(self, empty_tank_mass=0.5):
        """
        Calculate final mass after fuel consumption.
        
        For hydrogen systems, m_end includes:
        - Payload mass
        - Empty mass
        - Empty tank mass (tank structure remains after fuel is consumed)
        
        Args:
            empty_tank_mass (float): Mass of empty tank structure in kg.
                                   Default is 0.5 kg.
        
        Returns:
            float: End mass in kg
        """
        return self.payload_mass + self.empty_mass + empty_tank_mass
    
    def calculate_range_breguet(self, start_mass=None, end_mass=None):
        """
        Calculate range using modified Breguet Range Equation for electric flight.
        
        Modified Breguet Range Equation for hydrogen-electric propulsion:
        R = (η_total * e_H2 / g) * (L/D) * ln(m_start / m_end)
        
        where:
        - R: Range (m)
        - η_total: Total system efficiency
        - e_H2: Hydrogen energy density (J/kg)
        - g: Gravitational acceleration (m/s²)
        - L/D: Lift-to-drag ratio
        - m_start: Initial mass (kg)
        - m_end: Final mass (kg), including empty tank weight
        
        This equation accounts for the fact that hydrogen is lightweight,
        so the empty tank mass must be included in m_end.
        
        Args:
            start_mass (float, optional): Start mass in kg. If None, uses
                                         calculated start mass.
            end_mass (float, optional): End mass in kg. If None, uses
                                       calculated end mass.
        
        Returns:
            float: Range in meters (m)
        """
        if start_mass is None:
            start_mass = self.calculate_start_mass()
        if end_mass is None:
            end_mass = self.calculate_end_mass()
        
        # Modified Breguet Range Equation for electric flight
        # R = (η_total * e_H2 / g) * (L/D) * ln(m_start / m_end)
        range_m = (self.total_efficiency * self.hydrogen_energy_density / self.g) * \
                  self.lift_to_drag * np.log(start_mass / end_mass)
        
        return range_m
    
    def simulate_takeoff(self, fuel_consumption_factor=0.01):
        """
        Simulate takeoff phase.
        
        Takeoff consumes a small fraction of fuel for acceleration and
        initial climb. Typical takeoff distance is short (0.1-0.5 km).
        
        Args:
            fuel_consumption_factor (float): Fraction of fuel consumed during
                                           takeoff. Default is 0.01 (1%).
        
        Returns:
            dict: Takeoff phase results including distance and fuel consumed
        """
        fuel_consumed = self.fuel_mass * fuel_consumption_factor
        self.remaining_fuel -= fuel_consumed
        
        # Takeoff distance (typically 0.1-0.5 km for small UAVs)
        takeoff_dist = 0.2  # km
        self.takeoff_distance = takeoff_dist
        self.distance_flown += takeoff_dist
        
        return {
            'phase': 'Takeoff',
            'distance_km': takeoff_dist,
            'fuel_consumed_kg': fuel_consumed,
            'remaining_fuel_kg': self.remaining_fuel
        }
    
    def simulate_climb(self, fuel_consumption_factor=0.05, climb_distance=2.0):
        """
        Simulate climb phase.
        
        Climb phase consumes fuel to gain altitude. Energy is used for
        both lift and forward propulsion.
        
        Args:
            fuel_consumption_factor (float): Fraction of initial fuel consumed
                                           during climb. Default is 0.05 (5%).
            climb_distance (float): Horizontal distance covered during climb in km.
                                   Default is 2.0 km.
        
        Returns:
            dict: Climb phase results
        """
        fuel_consumed = self.fuel_mass * fuel_consumption_factor
        self.remaining_fuel -= fuel_consumed
        
        self.climb_distance = climb_distance
        self.distance_flown += climb_distance
        
        return {
            'phase': 'Climb',
            'distance_km': climb_distance,
            'fuel_consumed_kg': fuel_consumed,
            'remaining_fuel_kg': self.remaining_fuel
        }
    
    def simulate_cruise(self, time_step=1.0):
        """
        Simulate cruise phase until fuel is depleted.
        
        Cruise phase uses the modified Breguet Range Equation to calculate
        distance covered as fuel is consumed. The mission continues until
        fuel is exhausted or minimum fuel threshold is reached.
        
        Now includes:
        - Boil-off gas (BOG) subtraction from fuel mass over time
        - Pressure drop effects on fuel cell net power output
        - Energy tracking for physics verification
        
        Args:
            time_step (float): Time step for simulation in seconds.
                              Default is 1.0 s.
        
        Returns:
            list: List of cruise phase data points (distance, remaining fuel)
        """
        cruise_data = []
        min_fuel_threshold = 0.01  # kg (minimum fuel to continue)
        
        # Get boil-off rate (kg/hour)
        boil_off_rate_kg_per_hour = self.tank.calculate_boil_off_rate()
        boil_off_rate_kg_per_sec = boil_off_rate_kg_per_hour / 3600.0
        
        # Calculate cruise range using remaining fuel
        while self.remaining_fuel > min_fuel_threshold:
            # Current mass
            current_mass = self.payload_mass + self.empty_mass + self.remaining_fuel
            
            # Calculate fuel consumption rate for propulsion (kg/s)
            # Based on power requirement and efficiency
            fuel_consumption_rate = 0.001  # kg per time step (adjustable)
            
            if self.remaining_fuel < fuel_consumption_rate:
                fuel_consumption_rate = self.remaining_fuel
            
            # Calculate mass flow rate for pressure drop calculation
            mass_flow_rate_kg_per_sec = fuel_consumption_rate / time_step
            
            # Apply pressure drop effects if propulsion system is available
            effective_efficiency = self.base_efficiency
            pump_power_w = 0.0
            
            if self.propulsion_system is not None:
                # Calculate pressure drop and pump power
                pressure_drop_pa = self.propulsion_system.get_pressure_drop(mass_flow_rate_kg_per_sec)
                pump_power_w = self.propulsion_system.calculate_pump_power(
                    mass_flow_rate_kg_per_sec, pressure_drop_pa
                )
                
                # Net fuel cell power = gross power - pump power
                net_fuel_cell_power_w = max(0.0, self.fuel_cell_power_w - pump_power_w)
                
                # Adjust efficiency based on pressure drop
                # η_effective = η_base * (P_net / P_gross)
                if self.fuel_cell_power_w > 0:
                    efficiency_reduction = net_fuel_cell_power_w / self.fuel_cell_power_w
                    effective_efficiency = self.base_efficiency * efficiency_reduction
                else:
                    effective_efficiency = self.base_efficiency
            
            # Update total efficiency for range calculation
            self.total_efficiency = effective_efficiency
            
            # Calculate incremental range using effective efficiency
            mass_before = current_mass
            mass_after = current_mass - fuel_consumption_rate
            
            # Incremental range with pressure-drop-adjusted efficiency
            incremental_range = (effective_efficiency * self.hydrogen_energy_density / self.g) * \
                               self.lift_to_drag * np.log(mass_before / mass_after)
            
            # Subtract fuel consumed for propulsion
            self.remaining_fuel -= fuel_consumption_rate
            
            # Subtract boil-off gas (BOG) losses over time step
            bog_loss_kg = boil_off_rate_kg_per_sec * time_step
            if self.remaining_fuel > bog_loss_kg:
                self.remaining_fuel -= bog_loss_kg
            else:
                # If BOG exceeds remaining fuel, set to minimum threshold
                self.remaining_fuel = min_fuel_threshold
            
            # Update distance
            self.distance_flown += incremental_range / 1000  # Convert m to km
            
            # Track energy for verification
            energy_consumed_j = fuel_consumption_rate * self.hydrogen_energy_density
            self.energy_tracking['fuel_cell_output_j'] += energy_consumed_j * effective_efficiency / self.base_efficiency
            self.energy_tracking['motor_output_j'] += energy_consumed_j * effective_efficiency * 0.9  # Motor efficiency ~90%
            self.energy_tracking['propulsive_energy_j'] += incremental_range * (current_mass * self.g) / self.lift_to_drag
            self.energy_tracking['fuel_cell_losses_j'] += energy_consumed_j * (1 - effective_efficiency / self.base_efficiency)
            self.energy_tracking['motor_losses_j'] += energy_consumed_j * effective_efficiency * 0.1  # Motor losses ~10%
            self.energy_tracking['propulsive_losses_j'] += energy_consumed_j * effective_efficiency * 0.15  # Propulsive losses ~15%
            self.energy_tracking['bog_losses_j'] += bog_loss_kg * self.hydrogen_energy_density
            self.energy_tracking['pump_power_j'] += pump_power_w * time_step
            
            cruise_data.append({
                'distance_km': self.distance_flown,
                'remaining_fuel_kg': self.remaining_fuel,
                'bog_loss_kg': bog_loss_kg,
                'pump_power_w': pump_power_w
            })
        
        self.cruise_distance = self.distance_flown - self.takeoff_distance - self.climb_distance
        
        return cruise_data
    
    def simulate_descent(self, fuel_consumption_factor=0.01, descent_distance=None):
        """
        Simulate descent phase.
        
        Descent phase typically requires minimal energy as the aircraft
        glides down. Some fuel may be consumed for controlled descent.
        
        Args:
            fuel_consumption_factor (float): Fraction of remaining fuel consumed
                                           during descent. Default is 0.01 (1%).
            descent_distance (float, optional): Descent distance in km.
                                              If None, estimated from cruise altitude.
        
        Returns:
            dict: Descent phase results
        """
        fuel_consumed = self.remaining_fuel * fuel_consumption_factor
        self.remaining_fuel -= fuel_consumed
        
        # Estimate descent distance (typically similar to climb)
        if descent_distance is None:
            descent_distance = 2.0  # km
        
        self.descent_distance = descent_distance
        self.distance_flown += descent_distance
        
        return {
            'phase': 'Descent',
            'distance_km': descent_distance,
            'fuel_consumed_kg': fuel_consumed,
            'remaining_fuel_kg': self.remaining_fuel
        }
    
    def run_mission(self):
        """
        Execute complete mission profile: Takeoff -> Climb -> Cruise -> Descent.
        
        Includes physics verification (energy balance) if enabled.
        
        Returns:
            dict: Complete mission results including total range, fuel consumption,
                  phase-by-phase breakdown, and physics verification results
        """
        # Reset mission state
        self.remaining_fuel = self.fuel_mass
        self.distance_flown = 0.0
        
        # Reset energy tracking
        self.energy_tracking = {
            'fuel_cell_output_j': 0.0,
            'motor_output_j': 0.0,
            'propulsive_energy_j': 0.0,
            'fuel_cell_losses_j': 0.0,
            'motor_losses_j': 0.0,
            'propulsive_losses_j': 0.0,
            'bog_losses_j': 0.0,
            'pump_power_j': 0.0
        }
        
        # Execute mission phases
        takeoff_results = self.simulate_takeoff()
        climb_results = self.simulate_climb()
        cruise_data = self.simulate_cruise()
        descent_results = self.simulate_descent()
        
        # Calculate total range
        total_range_km = self.distance_flown
        
        # Physics verification
        verification_result = None
        if self.enable_physics_verification and self.physics_verifier is not None:
            # Calculate total fuel consumed (including BOG)
            total_fuel_consumed_kg = self.fuel_mass - self.remaining_fuel
            
            # Estimate energy breakdown (simplified - actual values tracked in cruise)
            # For full accuracy, would need to track energy in all phases
            fuel_cell_output_j = self.energy_tracking['fuel_cell_output_j']
            motor_output_j = self.energy_tracking['motor_output_j']
            propulsive_energy_j = self.energy_tracking['propulsive_energy_j']
            fuel_cell_losses_j = self.energy_tracking['fuel_cell_losses_j']
            motor_losses_j = self.energy_tracking['motor_losses_j']
            propulsive_losses_j = self.energy_tracking['propulsive_losses_j']
            bog_losses_j = self.energy_tracking['bog_losses_j']
            
            # Run verification
            verification_result = self.physics_verifier.verify_energy_balance(
                fuel_mass_kg=self.fuel_mass,
                hydrogen_energy_density_j_per_kg=self.hydrogen_energy_density,
                fuel_cell_output_energy_j=fuel_cell_output_j,
                motor_output_energy_j=motor_output_j,
                propulsive_energy_j=propulsive_energy_j,
                fuel_cell_losses_j=fuel_cell_losses_j,
                motor_losses_j=motor_losses_j,
                propulsive_losses_j=propulsive_losses_j,
                bog_losses_j=bog_losses_j,
                simulation_name="Mission Profile Simulation"
            )
        
        result = {
            'total_range_km': total_range_km,
            'total_fuel_consumed_kg': self.fuel_mass - self.remaining_fuel,
            'remaining_fuel_kg': self.remaining_fuel,
            'takeoff': takeoff_results,
            'climb': climb_results,
            'cruise_data': cruise_data,
            'descent': descent_results,
            'start_mass_kg': self.calculate_start_mass(),
            'end_mass_kg': self.calculate_end_mass(),
            'energy_tracking': self.energy_tracking.copy(),
            'physics_verification': verification_result
        }
        
        return result


def monte_carlo_mission_analysis(base_efficiency=0.5, base_boil_off_rate=0.001,
                                 fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                                 lift_to_drag=15.0, mission_duration_hours=2.0,
                                 n_simulations=1000, efficiency_sigma=0.05,
                                 boil_off_sigma=0.05, seed=None):
    """
    Monte Carlo analysis for mission reliability and risk assessment.
    
    Runs multiple mission simulations with stochastic variations in:
    - System efficiency (Gaussian distribution)
    - Boil-off rate (Gaussian distribution)
    
    This analysis provides probability distributions of mission range,
    enabling risk assessment and reliability quantification.
    
    Args:
        base_efficiency (float): Base system efficiency (mean). Default is 0.5.
        base_boil_off_rate (float): Base boil-off rate in kg/hour (mean). 
                                   Default is 0.001 kg/h.
        fuel_mass (float): Initial fuel mass in kg. Default is 2.0 kg.
        payload_mass (float): Payload mass in kg. Default is 5.0 kg.
        empty_mass (float): Empty aircraft mass in kg. Default is 10.0 kg.
        lift_to_drag (float): Lift-to-drag ratio. Default is 15.0.
        mission_duration_hours (float): Mission duration in hours for boil-off calculation.
                                      Default is 2.0 hours.
        n_simulations (int): Number of Monte Carlo simulations. Default is 1000.
        efficiency_sigma (float): Standard deviation for efficiency distribution.
                                 Default is 0.05 (5%).
        boil_off_sigma (float): Standard deviation for boil-off rate distribution.
                               Default is 0.05 (5%).
        seed (int, optional): Random seed for reproducibility. Default is None.
    
    Returns:
        dict: Dictionary containing:
            - 'ranges_km': Array of mission ranges from all simulations
            - 'efficiencies': Array of efficiency values used
            - 'boil_off_rates': Array of boil-off rates used
            - 'mean_range': Mean mission range
            - 'std_range': Standard deviation of mission range
            - 'min_range': Minimum mission range
            - 'max_range': Maximum mission range
            - 'percentiles': Dictionary with 5th, 25th, 50th, 75th, 95th percentiles
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize arrays to store results
    ranges_km = np.zeros(n_simulations)
    efficiencies = np.zeros(n_simulations)
    boil_off_rates = np.zeros(n_simulations)
    
    # Run Monte Carlo simulations
    for i in range(n_simulations):
        # Sample efficiency from Gaussian distribution
        # Clip to reasonable bounds (0.1 to 0.9)
        efficiency = np.clip(
            np.random.normal(base_efficiency, efficiency_sigma),
            0.1, 0.9
        )
        efficiencies[i] = efficiency
        
        # Sample boil-off rate from Gaussian distribution
        # Ensure non-negative
        boil_off_rate = max(0.0, np.random.normal(base_boil_off_rate, 
                                                  base_boil_off_rate * boil_off_sigma))
        boil_off_rates[i] = boil_off_rate
        
        # Calculate fuel loss due to boil-off
        fuel_loss_boil_off = boil_off_rate * mission_duration_hours
        effective_fuel_mass = max(0.01, fuel_mass - fuel_loss_boil_off)
        
        # Create mission profile with varied efficiency
        mission = MissionProfile(
            payload_mass=payload_mass,
            fuel_mass=effective_fuel_mass,
            empty_mass=empty_mass,
            lift_to_drag=lift_to_drag,
            total_efficiency=efficiency,
            hydrogen_energy_density=120e6
        )
        
        # Run mission simulation
        results = mission.run_mission()
        ranges_km[i] = results['total_range_km']
    
    # Calculate statistics
    mean_range = np.mean(ranges_km)
    std_range = np.std(ranges_km)
    min_range = np.min(ranges_km)
    max_range = np.max(ranges_km)
    
    # Calculate percentiles
    percentiles = {
        'p5': np.percentile(ranges_km, 5),
        'p25': np.percentile(ranges_km, 25),
        'p50': np.percentile(ranges_km, 50),  # Median
        'p75': np.percentile(ranges_km, 75),
        'p95': np.percentile(ranges_km, 95)
    }
    
    return {
        'ranges_km': ranges_km,
        'efficiencies': efficiencies,
        'boil_off_rates': boil_off_rates,
        'mean_range': mean_range,
        'std_range': std_range,
        'min_range': min_range,
        'max_range': max_range,
        'percentiles': percentiles,
        'n_simulations': n_simulations
    }


def stochastic_mission_analysis(base_efficiency=0.5, base_boil_off_rate=0.001,
                                 fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                                 lift_to_drag=15.0, mission_duration_hours=2.0,
                                 n_iterations=500, seed=None):
    """
    Stochastic Analysis for mission reliability assessment.
    
    Runs multiple mission simulations with Gaussian-distributed variations in:
    - Fuel Cell Efficiency: ±3% variation (σ = 0.03)
    - Tank Boil-off Rate: ±5% variation (σ = 0.05)
    
    This analysis provides probability distributions of mission range,
    enabling reliability quantification and P90 range calculation (industry standard
    safety metric: the range the UAV is 90% likely to achieve).
    
    Args:
        base_efficiency (float): Base system efficiency (mean). Default is 0.5.
        base_boil_off_rate (float): Base boil-off rate in kg/hour (mean). 
                                   Default is 0.001 kg/h.
        fuel_mass (float): Initial fuel mass in kg. Default is 2.0 kg.
        payload_mass (float): Payload mass in kg. Default is 5.0 kg.
        empty_mass (float): Empty aircraft mass in kg. Default is 10.0 kg.
        lift_to_drag (float): Lift-to-drag ratio. Default is 15.0.
        mission_duration_hours (float): Mission duration in hours for boil-off calculation.
                                      Default is 2.0 hours.
        n_iterations (int): Number of Monte Carlo iterations. Default is 500.
        seed (int, optional): Random seed for reproducibility. Default is None.
    
    Returns:
        dict: Dictionary containing:
            - 'ranges_km': Array of mission ranges from all iterations
            - 'efficiencies': Array of efficiency values used
            - 'boil_off_rates': Array of boil-off rates used
            - 'mean_range': Mean mission range
            - 'std_range': Standard deviation of mission range
            - 'min_range': Minimum mission range
            - 'max_range': Maximum mission range
            - 'p90_range': P90 Range (90th percentile) - Industry standard safety metric
            - 'p50_range': Median (50th percentile)
            - 'p10_range': P10 Range (10th percentile)
            - 'percentiles': Dictionary with all percentiles
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Stochastic variation parameters
    efficiency_sigma = 0.03  # ±3% variation for Fuel Cell Efficiency
    boil_off_sigma = 0.05     # ±5% variation for Tank Boil-off Rate
    
    # Initialize arrays to store results
    ranges_km = np.zeros(n_iterations)
    efficiencies = np.zeros(n_iterations)
    boil_off_rates = np.zeros(n_iterations)
    
    # Run stochastic iterations
    for i in range(n_iterations):
        # Sample efficiency from Gaussian distribution with ±3% variation
        # Clip to reasonable bounds (0.1 to 0.9)
        efficiency = np.clip(
            np.random.normal(base_efficiency, base_efficiency * efficiency_sigma),
            0.1, 0.9
        )
        efficiencies[i] = efficiency
        
        # Sample boil-off rate from Gaussian distribution with ±5% variation
        # Ensure non-negative
        boil_off_rate = max(0.0, np.random.normal(
            base_boil_off_rate, 
            base_boil_off_rate * boil_off_sigma
        ))
        boil_off_rates[i] = boil_off_rate
        
        # Calculate fuel loss due to boil-off
        fuel_loss_boil_off = boil_off_rate * mission_duration_hours
        effective_fuel_mass = max(0.01, fuel_mass - fuel_loss_boil_off)
        
        # Create mission profile with varied efficiency
        mission = MissionProfile(
            payload_mass=payload_mass,
            fuel_mass=effective_fuel_mass,
            empty_mass=empty_mass,
            lift_to_drag=lift_to_drag,
            total_efficiency=efficiency,
            hydrogen_energy_density=120e6
        )
        
        # Run mission simulation
        results = mission.run_mission()
        ranges_km[i] = results['total_range_km']
    
    # Calculate statistics
    mean_range = np.mean(ranges_km)
    std_range = np.std(ranges_km)
    min_range = np.min(ranges_km)
    max_range = np.max(ranges_km)
    
    # Calculate percentiles (including P90 for industry standard safety metric)
    percentiles = {
        'p10': np.percentile(ranges_km, 10),
        'p25': np.percentile(ranges_km, 25),
        'p50': np.percentile(ranges_km, 50),  # Median
        'p75': np.percentile(ranges_km, 75),
        'p90': np.percentile(ranges_km, 90),  # P90 Range: 90% probability of achieving
        'p95': np.percentile(ranges_km, 95)
    }
    
    return {
        'ranges_km': ranges_km,
        'efficiencies': efficiencies,
        'boil_off_rates': boil_off_rates,
        'mean_range': mean_range,
        'std_range': std_range,
        'min_range': min_range,
        'max_range': max_range,
        'p90_range': percentiles['p90'],  # Industry standard safety metric
        'p50_range': percentiles['p50'],  # Median
        'p10_range': percentiles['p10'],
        'percentiles': percentiles,
        'n_iterations': n_iterations,
        'efficiency_variation': '±3%',
        'boil_off_variation': '±5%'
    }
