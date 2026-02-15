"""
Propulsion System Module for Hydrogen-Electric UAV Propulsion Simulator

This module implements CFD-based fluid dynamics analysis for hydrogen injection systems,
including pressure drop calculations and parasitic power losses from pumping systems.
Integrates real-world fluid dynamics into the 1D simulation framework.

Author: Senior Aerospace Propulsion Engineer
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d


class PropulsionSystem:
    """
    Propulsion system model incorporating CFD-based pressure drop analysis.
    
    Implements:
    - Injector pressure drop lookup from CFD data (CSV)
    - Parasitic power calculation for fuel pumps
    - Efficiency reduction based on pressure drop
    - Real-world fluid dynamics integration
    
    Critical for accurate system efficiency calculations where pump power
    directly impacts overall propulsion system performance.
    """
    
    def __init__(self, lookup_table_path=None):
        """
        Initialize propulsion system with pressure drop lookup table.
        
        Args:
            lookup_table_path (str, optional): Path to CSV file containing
                                              pressure drop vs mass flow rate data.
                                              If None, uses default lookup table.
        """
        self.lookup_table_path = lookup_table_path
        self.pressure_drop_data = None
        self.interpolator = None
        
        # Pump efficiency parameters (typical for aerospace cryogenic pumps)
        self.pump_efficiency = 0.65  # Pump mechanical efficiency (65%)
        self.pump_motor_efficiency = 0.85  # Motor efficiency (85%)
        self.combined_pump_efficiency = self.pump_efficiency * self.pump_motor_efficiency
        
        # Hydrogen properties
        self.h2_density = 70.8  # kg/m³ (liquid hydrogen at 20K)
        
        # Load lookup table
        self._load_lookup_table()
    
    def _load_lookup_table(self):
        """
        Load pressure drop lookup table from CSV file.
        
        Expected CSV format:
        mass_flow_rate_kg_per_s, pressure_drop_Pa
        0.001, 5000
        0.002, 12000
        ...
        
        If file not found, creates default lookup table based on typical
        injector characteristics (Darcy-Weisbach equation).
        """
        if self.lookup_table_path is None:
            # Default path: data/injector_pressure_drop.csv (relative path for Streamlit Cloud)
            default_path = Path(__file__).parent.parent / 'data' / 'injector_pressure_drop.csv'
            self.lookup_table_path = str(default_path)
        
        # Convert to Path object, handling both string and Path inputs
        if isinstance(self.lookup_table_path, str):
            # Handle relative paths for Streamlit Cloud compatibility
            if not Path(self.lookup_table_path).is_absolute():
                # Relative path - resolve relative to project root
                lookup_path = Path(__file__).parent.parent / self.lookup_table_path
            else:
                lookup_path = Path(self.lookup_table_path)
        else:
            lookup_path = Path(self.lookup_table_path)
        
        try:
            if lookup_path.exists():
                # Load from CSV
                self.pressure_drop_data = pd.read_csv(lookup_path)
                
                # Validate columns
                required_cols = ['mass_flow_rate_kg_per_s', 'pressure_drop_Pa']
                if not all(col in self.pressure_drop_data.columns for col in required_cols):
                    raise ValueError(f"CSV must contain columns: {required_cols}")
                
                # Sort by mass flow rate for interpolation
                self.pressure_drop_data = self.pressure_drop_data.sort_values('mass_flow_rate_kg_per_s')
                
                # Create interpolator (linear interpolation with extrapolation)
                self.interpolator = interp1d(
                    self.pressure_drop_data['mass_flow_rate_kg_per_s'].values,
                    self.pressure_drop_data['pressure_drop_Pa'].values,
                    kind='linear',
                    fill_value='extrapolate',
                    bounds_error=False
                )
                
                print(f"Loaded pressure drop lookup table: {lookup_path}")
            else:
                # Create default lookup table based on typical injector characteristics
                print(f"Lookup table not found at {lookup_path}")
                print("Creating default lookup table based on typical injector characteristics...")
                self._create_default_lookup_table()
                
        except Exception as e:
            print(f"Warning: Error loading lookup table: {e}")
            print("Using default lookup table...")
            self._create_default_lookup_table()
    
    def _create_default_lookup_table(self):
        """
        Create default pressure drop lookup table based on typical injector characteristics.
        
        Uses Darcy-Weisbach equation approximation:
        ΔP = f * (L/D) * (ρ * v²) / 2
        
        For typical aerospace injectors with:
        - Flow coefficient: Cv ≈ 0.6-0.8
        - Injector diameter: 0.5-2.0 mm
        - Typical pressure drops: 0.01-0.5 MPa
        """
        # Generate mass flow rates (kg/s) - typical range for small UAV
        mass_flow_rates = np.array([
            0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005,
            0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.03
        ])
        
        # Calculate pressure drops using simplified injector model
        # ΔP ≈ K * (m_dot²) / (ρ * A²) where K is loss coefficient
        # For typical injector: K ≈ 2-5, A ≈ π * (0.001 m)²
        
        injector_area = np.pi * (0.001)**2  # m² (1mm diameter injector)
        loss_coefficient = 3.0  # Typical for aerospace injectors
        
        # Pressure drop: ΔP = K * (m_dot²) / (2 * ρ * A²)
        pressure_drops = loss_coefficient * (mass_flow_rates**2) / (2 * self.h2_density * injector_area**2)
        
        # Convert to Pascals and add base pressure drop
        pressure_drops = pressure_drops * 1e5  # Convert to Pa (typical range: 10-500 kPa)
        pressure_drops = pressure_drops + 10000  # Base pressure drop (10 kPa)
        
        # Create DataFrame
        self.pressure_drop_data = pd.DataFrame({
            'mass_flow_rate_kg_per_s': mass_flow_rates,
            'pressure_drop_Pa': pressure_drops
        })
        
        # Create interpolator
        self.interpolator = interp1d(
            self.pressure_drop_data['mass_flow_rate_kg_per_s'].values,
            self.pressure_drop_data['pressure_drop_Pa'].values,
            kind='linear',
            fill_value='extrapolate',
            bounds_error=False
        )
        
        print("Default lookup table created with typical injector characteristics")
    
    def get_pressure_drop(self, mass_flow_rate):
        """
        Get pressure drop for given mass flow rate using lookup table.
        
        Uses interpolation/extrapolation from CFD lookup table data.
        This represents the pressure drop across the injector system,
        which is critical for pump sizing and power requirements.
        
        Args:
            mass_flow_rate (float): Mass flow rate in kg/s
        
        Returns:
            float: Pressure drop in Pascals (Pa)
        """
        if self.interpolator is None:
            raise ValueError("Lookup table not initialized. Call _load_lookup_table() first.")
        
        # Ensure non-negative
        mass_flow_rate = max(0.0, mass_flow_rate)
        
        # Interpolate/extrapolate pressure drop
        pressure_drop = float(self.interpolator(mass_flow_rate))
        
        # Ensure non-negative
        return max(0.0, pressure_drop)
    
    def calculate_pump_power(self, mass_flow_rate, pressure_drop=None):
        """
        Calculate parasitic power required for fuel pump.
        
        Pump power is calculated using:
        P_pump = (m_dot * ΔP) / (ρ * η_pump)
        
        where:
        - m_dot: Mass flow rate (kg/s)
        - ΔP: Pressure drop (Pa)
        - ρ: Fluid density (kg/m³)
        - η_pump: Combined pump efficiency
        
        This parasitic power directly reduces system efficiency.
        
        Args:
            mass_flow_rate (float): Mass flow rate in kg/s
            pressure_drop (float, optional): Pressure drop in Pa.
                                           If None, calculated from lookup table.
        
        Returns:
            float: Pump power in Watts (W)
        """
        if pressure_drop is None:
            pressure_drop = self.get_pressure_drop(mass_flow_rate)
        
        # Pump power: P = (m_dot * ΔP) / (ρ * η)
        # For incompressible flow (liquid hydrogen)
        pump_power = (mass_flow_rate * pressure_drop) / (self.h2_density * self.combined_pump_efficiency)
        
        return pump_power
    
    def calculate_efficiency_reduction(self, mass_flow_rate, fuel_cell_power, pressure_drop=None):
        """
        Calculate efficiency reduction due to parasitic pump power.
        
        The pump power represents parasitic losses that reduce NET fuel cell power output.
        The pressure drop (ΔP) from CFD lookup table correctly subtracts from fuel cell's
        net power output, reducing effective system efficiency.
        
        Net Power Calculation:
        P_net = P_fuel_cell - P_pump
        
        Efficiency reduction is calculated as:
        η_effective = η_base * (P_net / P_fuel_cell)
        
        This integrates real-world fluid dynamics (pressure drops) into
        the 1D system efficiency model.
        
        Args:
            mass_flow_rate (float): Mass flow rate in kg/s
            fuel_cell_power (float): Fuel cell GROSS power output in Watts (W)
            pressure_drop (float, optional): Pressure drop in Pa.
                                           If None, calculated from lookup table.
        
        Returns:
            tuple: (efficiency_reduction_factor, pump_power_W, net_power_W)
                   - efficiency_reduction_factor: Multiplier for base efficiency (0-1)
                   - pump_power_W: Parasitic pump power (W) - SUBTRACTS from fuel cell power
                   - net_power_W: Net fuel cell power after pump losses (W)
        """
        # Calculate pump power (parasitic loss)
        pump_power = self.calculate_pump_power(mass_flow_rate, pressure_drop)
        
        # Net fuel cell power = Gross power - Pump power
        # Pressure drop correctly subtracts from fuel cell net power output
        net_power = max(0.0, fuel_cell_power - pump_power)
        
        # Efficiency reduction factor
        # η_effective = η_base * (P_net / P_gross)
        if fuel_cell_power > 0:
            efficiency_reduction_factor = net_power / fuel_cell_power
        else:
            efficiency_reduction_factor = 1.0
        
        return efficiency_reduction_factor, pump_power, net_power
    
    def get_efficiency_with_pressure_drop(self, base_efficiency, mass_flow_rate, fuel_cell_power):
        """
        Get system efficiency accounting for pressure drop losses.
        
        This is the main function for integrating CFD-based pressure drop
        analysis into the system efficiency model. It:
        1. Looks up pressure drop (ΔP) from CFD data (CSV)
        2. Calculates parasitic pump power (subtracts from fuel cell net power)
        3. Reduces base efficiency accordingly
        
        The pressure drop correctly subtracts from fuel cell's net power output,
        reducing effective system efficiency.
        
        Args:
            base_efficiency (float): Base system efficiency (0-1)
            mass_flow_rate (float): Mass flow rate in kg/s
            fuel_cell_power (float): Fuel cell GROSS power output in Watts (W)
        
        Returns:
            float: Adjusted efficiency accounting for pressure drop losses (0-1)
        """
        efficiency_reduction_factor, pump_power, net_power = self.calculate_efficiency_reduction(
            mass_flow_rate, fuel_cell_power
        )
        
        # Adjusted efficiency
        # η_effective = η_base * (P_net / P_gross)
        adjusted_efficiency = base_efficiency * efficiency_reduction_factor
        
        return adjusted_efficiency
    
    def get_pressure_drop_data(self):
        """
        Get the loaded pressure drop lookup table data.
        
        Returns:
            pandas.DataFrame: Pressure drop lookup table
        """
        return self.pressure_drop_data.copy() if self.pressure_drop_data is not None else None
    
    def save_lookup_table(self, output_path):
        """
        Save current lookup table to CSV file.
        
        Useful for exporting default lookup table or modified data.
        
        Args:
            output_path (str): Path to save CSV file (relative or absolute)
        """
        if self.pressure_drop_data is not None:
            # Handle relative paths for Streamlit Cloud compatibility
            if isinstance(output_path, str) and not Path(output_path).is_absolute():
                # Relative path - resolve relative to project root
                output_file = Path(__file__).parent.parent / output_path
            else:
                output_file = Path(output_path)
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            self.pressure_drop_data.to_csv(output_file, index=False)
            print(f"Lookup table saved to: {output_file}")
        else:
            raise ValueError("No lookup table data available to save")
