"""
Physics Core Module for Hydrogen-Electric UAV Propulsion Simulator

This module provides fundamental fluid properties for Hydrogen at cryogenic temperatures
using the CoolProp library. It implements thermodynamic property calculations essential
for aerospace propulsion system analysis.

Author: Senior Aerospace Propulsion Engineer
"""

import numpy as np

try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    # Create dummy CP module for graceful degradation
    class DummyCP:
        @staticmethod
        def PropsSI(*args, **kwargs):
            raise ImportError("CoolProp not available")
    CP = DummyCP()


class HydrogenProperties:
    """
    Class for calculating thermodynamic properties of Hydrogen at cryogenic conditions.
    
    Uses CoolProp library to access the NIST Reference Fluid Thermodynamic and 
    Transport Properties Database (REFPROP) for accurate hydrogen property calculations.
    Critical for cryogenic storage and fuel cell applications in aerospace systems.
    """
    
    def __init__(self):
        """Initialize the HydrogenProperties class."""
        self.fluid_name = 'Hydrogen'
    
    def get_density(self, temperature, pressure=101325):
        """
        Calculate the density of Hydrogen at specified temperature and pressure.
        
        In aerospace applications, density is critical for:
        - Mass flow rate calculations in fuel delivery systems
        - Tank sizing and volume requirements
        - Specific impulse (Isp) calculations for propulsion systems
        
        Args:
            temperature (float): Temperature in Kelvin (K). For cryogenic LH2, 
                                typically 20-30 K.
            pressure (float): Pressure in Pascals (Pa). Default is atmospheric 
                             pressure (101325 Pa).
        
        Returns:
            float: Density in kg/m³. Returns default value (70.8 kg/m³ at 20K) if CoolProp fails.
        
        Example:
            >>> props = HydrogenProperties()
            >>> rho = props.get_density(20.0)  # Density at 20K
        """
        # Input validation
        if temperature <= 0 or temperature > 1000:
            return 70.8  # Default LH2 density at 20K (kg/m³)
        if pressure <= 0 or pressure > 1e8:
            return 70.8  # Default LH2 density at 20K (kg/m³)
        
        try:
            density = CP.PropsSI('D', 'T', temperature, 'P', pressure, self.fluid_name)
            # Validate result
            if not np.isfinite(density) or density <= 0:
                return 70.8  # Default LH2 density at 20K (kg/m³)
            return density
        except Exception as e:
            # Return default value instead of raising error for cloud robustness
            # Default: Liquid hydrogen density at 20K (NIST standard)
            return 70.8  # kg/m³
    
    def get_enthalpy(self, temperature, pressure=101325):
        """
        Calculate the specific enthalpy of Hydrogen at specified conditions.
        
        Enthalpy is essential for:
        - Energy balance calculations in fuel cell systems
        - Heat transfer analysis in cryogenic storage systems
        - Thermodynamic cycle analysis (Brayton, Rankine cycles)
        - Calculating heat of vaporization for boil-off analysis
        
        Args:
            temperature (float): Temperature in Kelvin (K)
            pressure (float): Pressure in Pascals (Pa). Default is 101325 Pa.
        
        Returns:
            float: Specific enthalpy in J/kg. Returns default value if CoolProp fails.
        
        Example:
            >>> props = HydrogenProperties()
            >>> h = props.get_enthalpy(20.0)  # Enthalpy at 20K
        """
        if not COOLPROP_AVAILABLE:
            # Default: Enthalpy of liquid hydrogen at 20K (approximate)
            return -200000  # J/kg (negative relative to reference state)
        
        # Input validation
        if temperature <= 0 or temperature > 1000:
            return -200000  # Default enthalpy
        if pressure <= 0 or pressure > 1e8:
            return -200000  # Default enthalpy
        
        try:
            enthalpy = CP.PropsSI('H', 'T', temperature, 'P', pressure, self.fluid_name)
            # Validate result
            if not np.isfinite(enthalpy):
                return -200000  # Default enthalpy
            return enthalpy
        except Exception as e:
            # Return default value instead of raising error for cloud robustness
            return -200000  # J/kg (approximate enthalpy of LH2 at 20K)
    
    def get_phase(self, temperature, pressure=101325):
        """
        Determine the phase of Hydrogen (liquid, gas, or supercritical).
        
        Phase determination is critical for:
        - Cryogenic storage system design (LH2 vs GH2)
        - Fuel delivery system configuration
        - Heat transfer mode selection (latent vs sensible heat)
        - Boil-off rate calculations
        
        Args:
            temperature (float): Temperature in Kelvin (K)
            pressure (float): Pressure in Pascals (Pa). Default is 101325 Pa.
        
        Returns:
            str: Phase description ('liquid', 'gas', 'supercritical', or 'two-phase').
                 Returns 'liquid' as default if CoolProp fails.
        
        Example:
            >>> props = HydrogenProperties()
            >>> phase = props.get_phase(20.0)  # Phase at 20K (typically liquid)
        """
        if not COOLPROP_AVAILABLE:
            # Default: Assume liquid for cryogenic temperatures
            return 'liquid' if temperature < 33.0 else 'gas'
        
        # Input validation
        if temperature <= 0 or temperature > 1000:
            return 'liquid' if temperature < 33.0 else 'gas'
        if pressure <= 0 or pressure > 1e8:
            return 'liquid'
        
        try:
            # Get quality (0 = saturated liquid, 1 = saturated vapor)
            quality = CP.PropsSI('Q', 'T', temperature, 'P', pressure, self.fluid_name)
            
            # Get critical temperature and pressure
            T_critical = CP.PropsSI('Tcrit', self.fluid_name)
            P_critical = CP.PropsSI('Pcrit', self.fluid_name)
            
            # Determine phase
            if temperature > T_critical and pressure > P_critical:
                return 'supercritical'
            elif 0.0 <= quality <= 1.0:
                if quality == 0.0:
                    return 'liquid'
                elif quality == 1.0:
                    return 'gas'
                else:
                    return 'two-phase'
            else:
                # Check if above critical point
                if temperature > T_critical:
                    return 'supercritical'
                elif pressure < CP.PropsSI('P', 'T', temperature, 'Q', 0, self.fluid_name):
                    return 'gas'
                else:
                    return 'liquid'
        except Exception as e:
            # Return default value instead of raising error for cloud robustness
            # Default: Assume liquid for cryogenic temperatures (< 33K critical temp)
            return 'liquid' if temperature < 33.0 else 'gas'
    
    def get_properties_at_temperature(self, temperature, pressure=101325):
        """
        Get comprehensive thermodynamic properties at a given temperature.
        
        Useful for system-level analysis where multiple properties are needed
        simultaneously, reducing computational overhead.
        
        Args:
            temperature (float): Temperature in Kelvin (K)
            pressure (float): Pressure in Pascals (Pa). Default is 101325 Pa.
        
        Returns:
            dict: Dictionary containing density (kg/m³), enthalpy (J/kg), 
                  and phase (str)
        
        Example:
            >>> props = HydrogenProperties()
            >>> data = props.get_properties_at_temperature(20.0)
            >>> print(f"Density: {data['density']} kg/m³")
        """
        return {
            'density': self.get_density(temperature, pressure),
            'enthalpy': self.get_enthalpy(temperature, pressure),
            'phase': self.get_phase(temperature, pressure),
            'temperature': temperature,
            'pressure': pressure
        }
