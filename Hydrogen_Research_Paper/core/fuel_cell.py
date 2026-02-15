"""
PEM Fuel Cell Module for Hydrogen-Electric UAV Propulsion Simulator

This module implements a Proton Exchange Membrane (PEM) Fuel Cell model based on
electrochemical principles, specifically the Nernst Equation and Butler-Volmer
kinetics. Essential for predicting fuel cell performance in aerospace applications.

Author: Senior Aerospace Propulsion Engineer
"""

import numpy as np


class PEMFuelCell:
    """
    Proton Exchange Membrane (PEM) Fuel Cell model for aerospace applications.
    
    Implements the fundamental electrochemical equations governing fuel cell operation:
    - Nernst Equation for open circuit voltage
    - Butler-Volmer equation for activation losses
    - Ohmic losses from ionic and electronic resistance
    - Concentration losses from mass transport limitations
    
    The cell voltage is calculated as:
    V_cell = E_ocv - η_act - η_ohm - η_conc
    """
    
    def __init__(self, E_ocv=1.2, temperature=353.15, pressure=101325):
        """
        Initialize PEM Fuel Cell parameters.
        
        Args:
            E_ocv (float): Open Circuit Voltage in Volts (V). Default is 1.2V,
                          typical for PEM fuel cells at standard conditions.
            temperature (float): Operating temperature in Kelvin (K). 
                               Default is 353.15 K (80°C), typical for PEMFC.
            pressure (float): Operating pressure in Pascals (Pa). 
                            Default is atmospheric pressure (101325 Pa).
        """
        self.E_ocv = E_ocv  # Open Circuit Voltage (V)
        self.temperature = temperature  # Operating temperature (K)
        self.pressure = pressure  # Operating pressure (Pa)
        
        # Physical constants
        self.R = 8.314  # Universal gas constant (J/(mol·K))
        self.F = 96485  # Faraday's constant (C/mol)
        
        # Electrochemical parameters (typical values for PEMFC)
        self.alpha_a = 0.5  # Anode transfer coefficient (dimensionless)
        self.alpha_c = 1.0  # Cathode transfer coefficient (dimensionless)
        self.i0 = 1e-3  # Exchange current density (A/m²)
        self.R_ohm = 0.2e-3  # Ohmic resistance (Ω·m²)
        self.i_L = 2.0  # Limiting current density (A/m²)
    
    def calculate_open_circuit_voltage(self, pH2=1.0, pO2=0.21):
        """
        Calculate the Open Circuit Voltage (OCV) using the Nernst Equation.
        
        The Nernst Equation relates the reversible cell potential to reactant
        concentrations and temperature. For a hydrogen-oxygen fuel cell:
        
        E_ocv = E° - (RT/2F) * ln(1/(pH2 * pO2^0.5))
        
        where E° is the standard cell potential (typically 1.23V at 25°C).
        
        In aerospace applications, OCV determines the maximum theoretical
        efficiency and energy density of the fuel cell system.
        
        Args:
            pH2 (float): Partial pressure of hydrogen (bar). Default is 1.0 bar.
            pO2 (float): Partial pressure of oxygen (bar). Default is 0.21 bar 
                         (atmospheric air).
        
        Returns:
            float: Open Circuit Voltage in Volts (V)
        """
        # Standard cell potential at reference conditions (1.23V at 25°C)
        E0 = 1.23
        
        # Nernst Equation: E = E° - (RT/2F) * ln(1/(pH2 * pO2^0.5))
        # For H2 + 0.5O2 -> H2O, n = 2 electrons transferred
        n = 2
        
        # Calculate the Nernst correction term
        nernst_term = (self.R * self.temperature) / (n * self.F) * np.log(1.0 / (pH2 * np.sqrt(pO2)))
        
        E_ocv = E0 - nernst_term
        
        return E_ocv
    
    def calculate_activation_loss(self, current_density):
        """
        Calculate activation overpotential using simplified Butler-Volmer equation.
        
        Activation losses occur due to the energy barrier for electrochemical
        reactions at the electrode surfaces. The Butler-Volmer equation describes
        the relationship between current density and activation overpotential:
        
        η_act = (RT/αF) * asinh(i / (2*i0))
        
        This simplified form assumes symmetric charge transfer coefficients and
        is valid for moderate current densities typical in aerospace applications.
        
        Args:
            current_density (float or np.ndarray): Current density in A/m²
        
        Returns:
            float or np.ndarray: Activation overpotential in Volts (V)
        """
        # Simplified Butler-Volmer form for activation losses
        # Using average transfer coefficient
        alpha_avg = (self.alpha_a + self.alpha_c) / 2
        
        # Activation overpotential: η_act = (RT/αF) * asinh(i / (2*i0))
        eta_act = (self.R * self.temperature) / (alpha_avg * self.F) * np.arcsinh(
            current_density / (2 * self.i0)
        )
        
        return eta_act
    
    def calculate_ohmic_loss(self, current_density):
        """
        Calculate ohmic overpotential from ionic and electronic resistance.
        
        Ohmic losses arise from:
        - Ionic resistance in the electrolyte (membrane)
        - Electronic resistance in electrodes and current collectors
        - Contact resistance at interfaces
        
        η_ohm = i * R_ohm
        
        In aerospace systems, minimizing ohmic losses is critical for high
        power density and efficiency.
        
        Args:
            current_density (float or np.ndarray): Current density in A/m²
        
        Returns:
            float or np.ndarray: Ohmic overpotential in Volts (V)
        """
        eta_ohm = current_density * self.R_ohm
        return eta_ohm
    
    def calculate_concentration_loss(self, current_density):
        """
        Calculate concentration overpotential from mass transport limitations.
        
        Concentration losses occur when reactant supply cannot keep up with
        consumption at high current densities. This is modeled as:
        
        η_conc = (RT/2F) * ln(1 - i/i_L)
        
        where i_L is the limiting current density.
        
        Critical for high-altitude aerospace applications where oxygen
        partial pressure is reduced.
        
        Args:
            current_density (float or np.ndarray): Current density in A/m²
        
        Returns:
            float or np.ndarray: Concentration overpotential in Volts (V)
        """
        # Avoid division by zero and log of negative numbers
        current_density = np.clip(current_density, 0, 0.99 * self.i_L)
        
        # Concentration overpotential
        eta_conc = (self.R * self.temperature) / (2 * self.F) * np.log(
            1 - current_density / self.i_L
        )
        
        return eta_conc
    
    def calculate_cell_voltage(self, current_density, pH2=1.0, pO2=0.21):
        """
        Calculate the cell voltage accounting for all losses.
        
        The cell voltage is the open circuit voltage minus all overpotentials:
        
        V_cell = E_ocv - η_act - η_ohm - η_conc
        
        This is the fundamental equation for fuel cell performance prediction
        in aerospace propulsion systems.
        
        Args:
            current_density (float or np.ndarray): Current density in A/m²
            pH2 (float): Partial pressure of hydrogen (bar). Default is 1.0 bar.
            pO2 (float): Partial pressure of oxygen (bar). Default is 0.21 bar.
        
        Returns:
            float or np.ndarray: Cell voltage in Volts (V)
        """
        # Calculate open circuit voltage
        E_ocv = self.calculate_open_circuit_voltage(pH2, pO2)
        
        # Calculate all losses
        eta_act = self.calculate_activation_loss(current_density)
        eta_ohm = self.calculate_ohmic_loss(current_density)
        eta_conc = self.calculate_concentration_loss(current_density)
        
        # Cell voltage = OCV - all losses
        V_cell = E_ocv - eta_act - eta_ohm - eta_conc
        
        return V_cell
    
    def calculate_power_density(self, current_density, pH2=1.0, pO2=0.21):
        """
        Calculate power density from current density and cell voltage.
        
        Power density (W/m²) = Current density (A/m²) × Cell voltage (V)
        
        Essential metric for aerospace applications where power-to-weight
        ratio is critical.
        
        Args:
            current_density (float or np.ndarray): Current density in A/m²
            pH2 (float): Partial pressure of hydrogen (bar). Default is 1.0 bar.
            pO2 (float): Partial pressure of oxygen (bar). Default is 0.21 bar.
        
        Returns:
            float or np.ndarray: Power density in W/m²
        """
        V_cell = self.calculate_cell_voltage(current_density, pH2, pO2)
        power_density = current_density * V_cell
        return power_density
