"""
LH2 Storage Tank Module for Hydrogen-Electric UAV Propulsion Simulator

This module models cryogenic liquid hydrogen (LH2) storage tanks with Multi-Layer
Insulation (MLI) systems. Implements heat-leak calculations using Fourier's Law
and boil-off rate predictions essential for aerospace propulsion system design.

Author: Senior Aerospace Propulsion Engineer
"""

import numpy as np


class LH2Tank:
    """
    Liquid Hydrogen (LH2) Storage Tank model for aerospace applications.
    
    Implements thermal analysis of cryogenic storage systems including:
    - Multi-Layer Insulation (MLI) heat transfer modeling
    - Boil-off rate calculations based on heat-leak
    - Tank geometry and insulation performance analysis
    
    Critical for mission planning where boil-off losses directly impact
    available fuel mass and system efficiency.
    """
    
    def __init__(self, tank_radius=0.5, tank_length=2.0, ambient_temp=293.15):
        """
        Initialize LH2 storage tank parameters.
        
        Args:
            tank_radius (float): Tank radius in meters (m). Default is 0.5 m.
            tank_length (float): Tank length in meters (m). Default is 2.0 m.
            ambient_temp (float): Ambient temperature in Kelvin (K). 
                                Default is 293.15 K (20°C).
        """
        self.tank_radius = tank_radius  # Tank radius (m)
        self.tank_length = tank_length  # Tank length (m)
        self.ambient_temp = ambient_temp  # Ambient temperature (K)
        
        # LH2 properties (at saturation, ~20K)
        self.lh2_temp = 20.0  # Liquid hydrogen temperature (K)
        self.h_fg = 448000  # Latent heat of vaporization (J/kg) at 20K
        
        # MLI properties (typical values for aerospace applications)
        self.mli_layers = 30  # Number of MLI layers
        self.mli_emissivity = 0.03  # Effective emissivity per layer
        self.mli_thermal_conductivity = 0.00002  # Effective thermal conductivity (W/(m·K))
        self.mli_thickness = 0.05  # MLI thickness (m)
    
    def calculate_surface_area(self):
        """
        Calculate the total surface area of the cylindrical tank.
        
        For a cylindrical tank with hemispherical ends:
        A = 2πr² (hemispheres) + 2πrL (cylinder)
        
        Surface area determines heat transfer rate in cryogenic systems.
        
        Returns:
            float: Total surface area in m²
        """
        # Hemispherical end caps area
        hemisphere_area = 2 * np.pi * self.tank_radius**2
        
        # Cylindrical body area
        cylinder_area = 2 * np.pi * self.tank_radius * self.tank_length
        
        total_area = hemisphere_area + cylinder_area
        return total_area
    
    def calculate_heat_leak_mli(self):
        """
        Calculate heat-leak through Multi-Layer Insulation using Fourier's Law.
        
        Fourier's Law for one-dimensional steady-state heat conduction:
        Q = -k * A * (dT/dx)
        
        For MLI systems, the effective thermal conductivity accounts for:
        - Radiation heat transfer between layers
        - Conduction through spacer materials
        - Gas conduction in residual vacuum
        
        The heat-leak rate is:
        Q = k_eff * A * (T_ambient - T_LH2) / t_MLI
        
        This is critical for predicting boil-off rates and thermal performance
        in aerospace cryogenic storage systems.
        
        Returns:
            float: Heat-leak rate in Watts (W)
        """
        # Calculate temperature difference
        delta_T = self.ambient_temp - self.lh2_temp
        
        # Calculate surface area
        surface_area = self.calculate_surface_area()
        
        # Fourier's Law: Q = k * A * (ΔT / t)
        # For MLI, effective thermal conductivity accounts for radiation and conduction
        heat_leak = (self.mli_thermal_conductivity * surface_area * delta_T) / self.mli_thickness
        
        return heat_leak
    
    def calculate_heat_leak_radiation(self):
        """
        Calculate radiative heat-leak component through MLI.
        
        Radiation heat transfer through MLI layers follows:
        Q_rad = ε_eff * σ * A * (T_amb^4 - T_LH2^4)
        
        where ε_eff = ε_layer^N for N layers in series.
        
        This component is significant in high-vacuum conditions typical
        of aerospace cryogenic systems.
        
        Returns:
            float: Radiative heat-leak rate in Watts (W)
        """
        # Stefan-Boltzmann constant
        sigma = 5.67e-8  # W/(m²·K⁴)
        
        # Effective emissivity for N layers: ε_eff = ε_layer^N
        epsilon_eff = self.mli_emissivity ** self.mli_layers
        
        # Calculate surface area
        surface_area = self.calculate_surface_area()
        
        # Radiative heat transfer: Q = ε_eff * σ * A * (T_amb^4 - T_LH2^4)
        heat_leak_rad = epsilon_eff * sigma * surface_area * (
            self.ambient_temp**4 - self.lh2_temp**4
        )
        
        return heat_leak_rad
    
    def calculate_total_heat_leak(self):
        """
        Calculate total heat-leak combining conduction and radiation.
        
        Total heat-leak is the sum of:
        - Conductive heat transfer through MLI (Fourier's Law)
        - Radiative heat transfer through MLI layers
        
        This determines the total thermal load on the cryogenic system.
        
        Returns:
            float: Total heat-leak rate in Watts (W)
        """
        # Conductive component (Fourier's Law)
        Q_cond = self.calculate_heat_leak_mli()
        
        # Radiative component
        Q_rad = self.calculate_heat_leak_radiation()
        
        # Total heat-leak
        total_heat_leak = Q_cond + Q_rad
        
        return total_heat_leak
    
    def calculate_boil_off_rate(self):
        """
        Calculate Boil-Off Rate (BOR) based on heat-leak.
        
        The boil-off rate is determined by the energy balance:
        Q_heat_leak = m_dot_BOR * h_fg
        
        where:
        - Q_heat_leak: Total heat-leak rate (W)
        - m_dot_BOR: Boil-off mass flow rate (kg/s)
        - h_fg: Latent heat of vaporization (J/kg)
        
        Solving for boil-off rate:
        m_dot_BOR = Q_heat_leak / h_fg
        
        This is converted to kg/hour for practical aerospace applications.
        
        Returns:
            float: Boil-off rate in kg/hour
        """
        # Calculate total heat-leak
        total_heat_leak = self.calculate_total_heat_leak()
        
        # Energy balance: Q = m_dot * h_fg
        # Solve for mass flow rate: m_dot = Q / h_fg
        boil_off_rate_kg_per_sec = total_heat_leak / self.h_fg
        
        # Convert to kg/hour
        boil_off_rate_kg_per_hour = boil_off_rate_kg_per_sec * 3600
        
        return boil_off_rate_kg_per_hour
    
    def calculate_gravimetric_index(self, fuel_mass, total_system_mass=None, 
                                   payload_mass=0, empty_aircraft_mass=0):
        """
        Calculate the Gravimetric Index for the LH2 storage system.
        
        The Gravimetric Index is a critical metric for aerospace applications:
        GI = m_fuel / m_total_system
        
        where m_total_system includes:
        - Fuel mass (m_fuel)
        - Tank structure mass
        - Payload mass
        - Empty aircraft mass (structure, systems, etc.)
        
        A higher gravimetric index indicates better fuel mass fraction,
        which directly impacts range and payload capacity. Modern LH2 UAV
        tanks typically target a gravimetric index of 0.25 (25%).
        
        Args:
            fuel_mass (float): Mass of LH2 fuel in kg
            total_system_mass (float, optional): Total system mass in kg.
                                               If None, calculated from other parameters.
            payload_mass (float): Payload mass in kg. Default is 0 kg.
            empty_aircraft_mass (float): Empty aircraft mass (excluding fuel/tank)
                                        in kg. Default is 0 kg.
        
        Returns:
            dict: Dictionary containing gravimetric index and component masses
        """
        # Calculate tank structure mass (simplified model)
        # Tank mass scales with volume and surface area
        surface_area = self.calculate_surface_area()
        tank_volume = np.pi * self.tank_radius**2 * self.tank_length + \
                      (4/3) * np.pi * self.tank_radius**3  # Cylinder + sphere
        
        # Tank structure mass estimation (typical: 50-100 kg/m³ for composite tanks)
        tank_density = 75.0  # kg/m³ (typical for composite cryogenic tanks)
        tank_structure_mass = tank_volume * tank_density
        
        # Calculate total system mass if not provided
        if total_system_mass is None:
            total_system_mass = fuel_mass + tank_structure_mass + payload_mass + empty_aircraft_mass
        
        # Calculate gravimetric index
        gravimetric_index = fuel_mass / total_system_mass if total_system_mass > 0 else 0.0
        
        # Target gravimetric index for modern LH2 UAV tanks
        target_gi = 0.25
        
        return {
            'gravimetric_index': gravimetric_index,
            'target_gravimetric_index': target_gi,
            'fuel_mass_kg': fuel_mass,
            'tank_structure_mass_kg': tank_structure_mass,
            'payload_mass_kg': payload_mass,
            'empty_aircraft_mass_kg': empty_aircraft_mass,
            'total_system_mass_kg': total_system_mass,
            'meets_target': gravimetric_index >= target_gi,
            'deviation_from_target': gravimetric_index - target_gi
        }
    
    def calculate_insulation_performance(self):
        """
        Calculate insulation performance metrics.
        
        Returns comprehensive thermal performance data including:
        - Total heat-leak rate
        - Boil-off rate
        - Heat flux
        - Effective thermal resistance
        
        Useful for system optimization and trade studies.
        
        Returns:
            dict: Dictionary containing thermal performance metrics
        """
        total_heat_leak = self.calculate_total_heat_leak()
        boil_off_rate = self.calculate_boil_off_rate()
        surface_area = self.calculate_surface_area()
        
        # Heat flux (W/m²)
        heat_flux = total_heat_leak / surface_area
        
        # Thermal resistance (K/W)
        delta_T = self.ambient_temp - self.lh2_temp
        thermal_resistance = delta_T / total_heat_leak if total_heat_leak > 0 else np.inf
        
        return {
            'total_heat_leak_W': total_heat_leak,
            'boil_off_rate_kg_per_hour': boil_off_rate,
            'heat_flux_W_per_m2': heat_flux,
            'thermal_resistance_K_per_W': thermal_resistance,
            'surface_area_m2': surface_area,
            'temperature_difference_K': delta_T
        }
