"""
Hydrogen-Electric UAV Propulsion Simulator - Interactive Trade-Study Dashboard

Professional engineering tool for analyzing hydrogen-electric propulsion systems
with interactive parameter adjustment and real-time visualization.

Author: Senior Aerospace Propulsion Engineer
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
import json
import re

# Import core modules
from core.fuel_cell import PEMFuelCell
from core.tank import LH2Tank
from core.mission import MissionProfile, monte_carlo_mission_analysis, stochastic_mission_analysis
from core.safety_compliance import SafetyCompliance
from core.assistant import PropulsionAssistant
from core.benchmarks import get_all_benchmarks, calculate_delta_analysis

# Import PDF generator (optional - requires reportlab)
try:
    from core.pdf_generator import generate_nasa_technical_memo
    PDF_GENERATOR_AVAILABLE = True
except ImportError:
    PDF_GENERATOR_AVAILABLE = False
    generate_nasa_technical_memo = None

# Set page config for professional appearance
st.set_page_config(
    page_title="Hydrogen-Electric UAV Propulsion Simulator",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use academic style for plots
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    try:
        plt.style.use('seaborn-paper')
    except OSError:
        plt.style.use('seaborn-whitegrid')

# Custom CSS for academic appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4788;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4788;
    }
    .stSlider > div > div > div {
        color: #1f4788;
    }
    </style>
""", unsafe_allow_html=True)

def calculate_tank_weight_from_thickness(insulation_thickness_mm):
    """
    Calculate tank weight based on insulation thickness.
    
    Thicker insulation increases tank weight but reduces heat leak.
    """
    insulation_thickness_m = insulation_thickness_mm / 1000.0  # Convert mm to m
    
    # Base tank structure mass (scales with insulation)
    # More insulation = more structural support needed
    base_tank_mass = 0.3  # kg (base structure)
    insulation_mass_per_mm = 0.01  # kg per mm of insulation thickness
    tank_weight = base_tank_mass + (insulation_thickness_mm * insulation_mass_per_mm)
    
    return tank_weight

def calculate_range_with_tank_weight(tank_weight, fuel_mass=2.0, payload_mass=5.0, 
                                     empty_mass=10.0, lift_to_drag=15.0, 
                                     total_efficiency=0.5, cruise_velocity=30.0):
    """
    Calculate mission range for given tank weight.
    
    Tank weight affects total system mass, which impacts range.
    
    Args:
        tank_weight: Tank weight in kg (must be >= 0)
        fuel_mass: Fuel mass in kg (must be > 0)
        payload_mass: Payload mass in kg (must be >= 0)
        empty_mass: Empty mass in kg (must be > 0)
        lift_to_drag: Lift-to-drag ratio (must be > 0)
        total_efficiency: Total system efficiency (must be > 0 and <= 1)
        cruise_velocity: Cruise velocity in m/s (must be > 0)
    
    Returns:
        float: Range in km
        
    Raises:
        ValueError: If input parameters would cause division by zero or invalid calculations
    """
    # Input validation
    if tank_weight < 0:
        raise ValueError("Tank weight must be >= 0 kg")
    if fuel_mass <= 0:
        raise ValueError("Fuel mass must be > 0 kg. Cannot calculate range with zero fuel.")
    if payload_mass < 0:
        raise ValueError("Payload mass must be >= 0 kg")
    if empty_mass <= 0:
        raise ValueError("Empty mass must be > 0 kg. Cannot calculate range with zero empty mass.")
    if lift_to_drag <= 0:
        raise ValueError("Lift-to-drag ratio must be > 0")
    if total_efficiency <= 0 or total_efficiency > 1:
        raise ValueError("Total efficiency must be > 0 and <= 1")
    if cruise_velocity <= 0:
        raise ValueError("Cruise velocity must be > 0 m/s")
    
    # Total empty mass includes tank weight
    total_empty_mass = empty_mass + tank_weight
    
    # Calculate start and end masses
    start_mass = payload_mass + fuel_mass + total_empty_mass
    end_mass = payload_mass + total_empty_mass + 0.1  # Small residual tank mass
    
    # Validate that end_mass < start_mass (required for log calculation)
    if end_mass >= start_mass:
        raise ValueError(f"End mass ({end_mass:.2f} kg) must be less than start mass ({start_mass:.2f} kg). "
                        f"Increase fuel mass or reduce empty mass/tank weight.")
    
    # Modified Breguet Range Equation
    hydrogen_energy_density = 120e6  # J/kg
    g = 9.81  # m/s²
    
    try:
        range_m = (total_efficiency * hydrogen_energy_density / g) * \
                  lift_to_drag * np.log(start_mass / end_mass)
        
        # Validate result
        if not np.isfinite(range_m) or range_m < 0:
            raise ValueError(f"Calculated range is invalid: {range_m}. Check input parameters.")
        
        return range_m / 1000  # Convert to km
    except (ZeroDivisionError, ValueError) as e:
        raise ValueError(f"Error calculating range: {str(e)}. "
                        f"Check that all input parameters are valid and non-zero where required.")

def calculate_range_with_altitude(insulation_thickness_mm, cruise_altitude_m, 
                                   fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                                   base_lift_to_drag=15.0, total_efficiency=0.5, 
                                   cruise_velocity=30.0, tank_radius=0.3, tank_length=1.5):
    """
    Calculate mission range considering both insulation thickness and cruise altitude.
    
    Includes input validation to prevent division by zero errors.
    """
    # Input validation
    if insulation_thickness_mm <= 0:
        raise ValueError("Insulation thickness must be > 0 mm")
    if cruise_altitude_m < 0:
        raise ValueError("Cruise altitude must be >= 0 m")
    if fuel_mass <= 0:
        raise ValueError("Fuel mass must be > 0 kg")
    if payload_mass < 0:
        raise ValueError("Payload mass must be >= 0 kg")
    if empty_mass <= 0:
        raise ValueError("Empty mass must be > 0 kg")
    if base_lift_to_drag <= 0:
        raise ValueError("Base lift-to-drag ratio must be > 0")
    if total_efficiency <= 0 or total_efficiency > 1:
        raise ValueError("Total efficiency must be > 0 and <= 1")
    if cruise_velocity <= 0:
        raise ValueError("Cruise velocity must be > 0 m/s")
    if tank_radius <= 0:
        raise ValueError("Tank radius must be > 0 m")
    if tank_length <= 0:
        raise ValueError("Tank length must be > 0 m")
    """
    Calculate mission range considering both insulation thickness and cruise altitude.
    
    Altitude effects:
    - Air density decreases with altitude, improving L/D ratio
    - Ambient temperature decreases with altitude, reducing boil-off rate
    
    Args:
        insulation_thickness_mm (float): MLI insulation thickness in mm
        cruise_altitude_m (float): Cruise altitude in meters
        Other parameters: Standard mission parameters
        
    Returns:
        float: Total range in km
    """
    # Calculate tank weight from insulation thickness
    tank_weight = calculate_tank_weight_from_thickness(insulation_thickness_mm)
    
    # Altitude effects on air density (exponential model)
    # Standard atmosphere: ρ = ρ₀ * exp(-h/H) where H ≈ 8400m
    scale_height = 8400.0  # m
    rho_sea_level = 1.225  # kg/m³
    air_density_ratio = np.exp(-cruise_altitude_m / scale_height)
    
    # L/D improves at altitude due to reduced drag
    # Simplified model: L/D scales approximately with sqrt(air_density_ratio)
    # Higher altitude = less drag = better L/D
    altitude_lift_to_drag = base_lift_to_drag * np.sqrt(1.0 / air_density_ratio)
    
    # Altitude effects on ambient temperature (standard atmosphere)
    # T = T₀ - L*h where L = 0.0065 K/m (lapse rate)
    lapse_rate = 0.0065  # K/m
    sea_level_temp = 288.15  # K (15°C)
    ambient_temp = sea_level_temp - lapse_rate * cruise_altitude_m
    ambient_temp = max(ambient_temp, 216.65)  # Limit to tropopause (~11km)
    
    # Calculate boil-off rate with altitude-dependent temperature
    tank = LH2Tank(tank_radius=tank_radius, tank_length=tank_length, 
                   ambient_temp=ambient_temp)
    tank.mli_thickness = insulation_thickness_mm / 1000.0  # Convert mm to m
    boil_off_rate_kg_per_hour = tank.calculate_boil_off_rate()
    
    # Estimate mission duration for boil-off calculation
    # Use base range estimate (iterative refinement possible but not necessary)
    total_empty_mass = empty_mass + tank_weight
    start_mass = payload_mass + fuel_mass + total_empty_mass
    end_mass = payload_mass + total_empty_mass + 0.1
    
    hydrogen_energy_density = 120e6  # J/kg
    g = 9.81  # m/s²
    
    base_range_m = (total_efficiency * hydrogen_energy_density / g) * \
                   altitude_lift_to_drag * np.log(start_mass / end_mass)
    
    # Mission duration (simplified)
    mission_duration_hours = (base_range_m / 1000.0) / (cruise_velocity * 3.6) if cruise_velocity > 0 else 2.0
    mission_duration_hours = max(mission_duration_hours, 0.1)  # Minimum 0.1 hours
    
    # Fuel loss due to boil-off
    fuel_loss_boil_off = boil_off_rate_kg_per_hour * mission_duration_hours
    effective_fuel_mass = max(0.1, fuel_mass - fuel_loss_boil_off)
    
    # Recalculate range with effective fuel mass (accounting for boil-off)
    effective_start_mass = payload_mass + effective_fuel_mass + total_empty_mass
    effective_end_mass = payload_mass + total_empty_mass + 0.1
    
    # Final range calculation with altitude-adjusted L/D
    range_m = (total_efficiency * hydrogen_energy_density / g) * \
              altitude_lift_to_drag * np.log(effective_start_mass / effective_end_mass)
    
    return range_m / 1000.0  # Convert to km

def calculate_liion_range(battery_mass, payload_mass=5.0, empty_mass=10.0,
                          lift_to_drag=15.0, total_efficiency=0.5):
    """
    Calculate range for Li-ion battery system.
    
    Args:
        battery_mass: Battery mass in kg
        Other parameters same as hydrogen system
    """
    liion_energy_density = 250  # Wh/kg = 900,000 J/kg
    liion_energy_density_jkg = liion_energy_density * 3600  # Convert to J/kg
    
    start_mass = payload_mass + battery_mass + empty_mass
    end_mass = payload_mass + empty_mass + 0.05  # Empty battery mass
    
    g = 9.81  # m/s²
    
    range_m = (total_efficiency * liion_energy_density_jkg / g) * \
              lift_to_drag * np.log(start_mass / end_mass)
    
    return range_m / 1000  # Convert to km

def calculate_fuel_mass_for_range(target_range_km, tank_weight, payload_mass=5.0, 
                                  empty_mass=10.0, lift_to_drag=15.0, 
                                  total_efficiency=0.5):
    """
    Calculate required fuel mass for a given range (reverse Breguet equation).
    
    Args:
        target_range_km: Desired range in kilometers
        tank_weight: Tank weight in kg
        Other parameters: System parameters
    
    Returns:
        float: Required fuel mass in kg
    """
    target_range_m = target_range_km * 1000  # Convert to meters
    hydrogen_energy_density = 120e6  # J/kg
    g = 9.81  # m/s²
    
    # Total empty mass includes tank weight
    total_empty_mass = empty_mass + tank_weight
    end_mass = payload_mass + total_empty_mass + 0.1  # Small residual
    
    # Reverse Breguet: R = (η * e / g) * (L/D) * ln(m_start / m_end)
    # Solve for m_start: m_start = m_end * exp(R * g / (η * e * L/D))
    start_mass = end_mass * np.exp(
        target_range_m * g / (total_efficiency * hydrogen_energy_density * lift_to_drag)
    )
    
    # Fuel mass = start_mass - payload - empty_mass - tank_weight
    fuel_mass = start_mass - payload_mass - total_empty_mass
    
    return max(0.0, fuel_mass)

def calculate_battery_mass_for_range(target_range_km, payload_mass=5.0, 
                                     empty_mass=10.0, lift_to_drag=15.0, 
                                     total_efficiency=0.5):
    """
    Calculate required battery mass for a given range (reverse Breguet equation).
    
    Args:
        target_range_km: Desired range in kilometers
        Other parameters: System parameters
    
    Returns:
        float: Required battery mass in kg
    """
    target_range_m = target_range_km * 1000  # Convert to meters
    liion_energy_density = 250  # Wh/kg
    liion_energy_density_jkg = liion_energy_density * 3600  # J/kg
    g = 9.81  # m/s²
    
    end_mass = payload_mass + empty_mass + 0.05  # Empty battery mass
    
    # Reverse Breguet: R = (η * e / g) * (L/D) * ln(m_start / m_end)
    # Solve for m_start: m_start = m_end * exp(R * g / (η * e * L/D))
    start_mass = end_mass * np.exp(
        target_range_m * g / (total_efficiency * liion_energy_density_jkg * lift_to_drag)
    )
    
    # Battery mass = start_mass - payload - empty_mass
    battery_mass = start_mass - payload_mass - empty_mass
    
    return max(0.0, battery_mass)

def calculate_h2_system_mass_for_range(target_range_km, tank_weight, payload_mass=5.0,
                                       empty_mass=10.0):
    """
    Calculate total system mass for hydrogen system at given range.
    
    Returns:
        float: Total system mass in kg
    """
    fuel_mass = calculate_fuel_mass_for_range(target_range_km, tank_weight, 
                                              payload_mass, empty_mass)
    total_mass = payload_mass + empty_mass + tank_weight + fuel_mass
    return total_mass

def calculate_liion_system_mass_for_range(target_range_km, payload_mass=5.0,
                                          empty_mass=10.0):
    """
    Calculate total system mass for Li-ion system at given range.
    
    Returns:
        float: Total system mass in kg
    """
    battery_mass = calculate_battery_mass_for_range(target_range_km, 
                                                    payload_mass, empty_mass)
    total_mass = payload_mass + empty_mass + battery_mass
    return total_mass

def find_break_even_range(tank_weight, payload_mass=5.0, empty_mass=10.0,
                          lift_to_drag=15.0, total_efficiency=0.5,
                          range_min=10.0, range_max=500.0, tolerance=0.1):
    """
    Find the break-even range where Hydrogen and Li-ion systems have equal mass.
    
    Uses binary search to find the intersection point.
    
    Returns:
        tuple: (break_even_range_km, system_mass_at_break_even_kg) or (None, None) if not found
    """
    # Binary search for break-even point
    low = range_min
    high = range_max
    
    for _ in range(50):  # Max iterations
        mid = (low + high) / 2
        
        h2_mass = calculate_h2_system_mass_for_range(mid, tank_weight, 
                                                     payload_mass, empty_mass)
        liion_mass = calculate_liion_system_mass_for_range(mid, payload_mass, empty_mass)
        
        diff = h2_mass - liion_mass
        
        if abs(diff) < tolerance:
            return mid, h2_mass
        
        if diff > 0:  # H2 heavier, need longer range
            low = mid
        else:  # H2 lighter, need shorter range
            high = mid
    
    return None, None

def validate_slider_input(value, min_val, max_val, param_name, must_be_positive=False):
    """
    Validate slider input values to prevent division by zero and invalid calculations.
    
    Args:
        value: The input value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        param_name: Name of the parameter for error messages
        must_be_positive: If True, value must be > 0 (not just >= 0)
    
    Returns:
        float: Validated value
        
    Raises:
        ValueError: If value is invalid
    """
    if value is None:
        raise ValueError(f"{param_name} cannot be None")
    
    if must_be_positive and value <= 0:
        raise ValueError(f"{param_name} must be > 0. Current value: {value}")
    
    if value < min_val or value > max_val:
        raise ValueError(f"{param_name} must be between {min_val} and {max_val}. Current value: {value}")
    
    if not np.isfinite(value):
        raise ValueError(f"{param_name} must be a finite number. Current value: {value}")
    
    return float(value)

def main():
    """Main Streamlit application."""
    
    # Header
    with st.container():
        st.markdown('<h1 class="main-header">Hydrogen-Electric UAV Propulsion Simulator</h1>', 
                    unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Interactive Trade-Study Dashboard for Aerospace Propulsion Analysis</p>', 
                    unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ System Parameters")
        st.markdown("---")
        
        # Insulation Thickness Slider
        st.subheader("Thermal Management")
        # Use AI-recommended value if available, otherwise use current value
        default_insulation = st.session_state.get('ai_insulation', st.session_state.get('current_insulation', 50.0))
        try:
            insulation_thickness_mm = st.slider(
                "Insulation Thickness (mm)",
                min_value=10.0,
                max_value=100.0,
                value=float(default_insulation),
                step=5.0,
                key="insulation_slider",
                help="Multi-Layer Insulation (MLI) thickness. Physics: Increasing this reduces boil-off heat leak (Q = kAΔT/t) but adds parasitic structural mass. Trade-off: Thicker MLI → Lower BOG → Higher tank weight → Reduced range."
            )
            insulation_thickness_mm = validate_slider_input(
                insulation_thickness_mm, 10.0, 100.0, "Insulation Thickness", must_be_positive=True
            )
            st.session_state['current_insulation'] = insulation_thickness_mm
        except ValueError as e:
            st.error(f"⚠️ Invalid input: {str(e)}")
            st.info("Using default value: 50.0 mm")
            insulation_thickness_mm = 50.0
            st.session_state['current_insulation'] = insulation_thickness_mm
        
        # Fuel Cell Stack Size Slider
        st.subheader("Power System")
        try:
            fuel_cell_power_kw = st.slider(
                "Fuel Cell Stack Size (kW)",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                help="Total fuel cell stack power output. Physics: Power = Voltage × Current. Higher power enables higher cruise speeds (P = F×v) but increases system mass (stack + cooling). Trade-off: More power → Higher speed capability → Higher mass → Reduced range at same speed."
            )
            fuel_cell_power_kw = validate_slider_input(
                fuel_cell_power_kw, 1.0, 20.0, "Fuel Cell Power", must_be_positive=True
            )
        except ValueError as e:
            st.error(f"⚠️ Invalid input: {str(e)}")
            st.info("Using default value: 5.0 kW")
            fuel_cell_power_kw = 5.0
        
        # Cruise Velocity Slider
        st.subheader("Flight Performance")
        # Use AI-recommended value if available, otherwise use current value
        default_velocity = st.session_state.get('ai_cruise_velocity', st.session_state.get('current_velocity', 30.0))
        try:
            cruise_velocity = st.slider(
                "Cruise Velocity (m/s)",
                min_value=15.0,
                max_value=50.0,
                value=float(default_velocity),
                step=1.0,
                key="cruise_velocity_slider",
                help="Cruise flight velocity. Physics: Drag power P_drag = D×v = (mg/L/D)×v. Higher velocity increases power requirements (P ∝ v³) and reduces range (R ∝ ln(m_start/m_end) / v). Trade-off: Higher v → Shorter mission time → Higher power → Reduced range."
            )
            cruise_velocity = validate_slider_input(
                cruise_velocity, 15.0, 50.0, "Cruise Velocity", must_be_positive=True
            )
            st.session_state['current_velocity'] = cruise_velocity
        except ValueError as e:
            st.error(f"⚠️ Invalid input: {str(e)}")
            st.info("Using default value: 30.0 m/s")
            cruise_velocity = 30.0
            st.session_state['current_velocity'] = cruise_velocity
        
        # Tank Radius Slider (for AI optimization)
        default_tank_radius = st.session_state.get('ai_tank_radius', st.session_state.get('tank_radius_m', 0.3))
        try:
            tank_radius_m = st.slider(
                "Tank Radius (m)",
                min_value=0.2,
                max_value=0.5,
                value=float(default_tank_radius),
                step=0.05,
                key="tank_radius_slider",
                help="Tank radius. Physics: Volume V = πr²L, Surface area A = 2πrL + 2πr². Larger radius improves volume-to-surface ratio (reduces heat leak per unit volume) but increases frontal area drag (D ∝ A_frontal). Trade-off: Larger r → Lower BOG rate → Higher drag → Reduced range."
            )
            tank_radius_m = validate_slider_input(
                tank_radius_m, 0.2, 0.5, "Tank Radius", must_be_positive=True
            )
            st.session_state['tank_radius_m'] = tank_radius_m
        except ValueError as e:
            st.error(f"⚠️ Invalid input: {str(e)}")
            st.info("Using default value: 0.3 m")
            tank_radius_m = 0.3
            st.session_state['tank_radius_m'] = tank_radius_m
        
        st.markdown("---")
        
        # Display calculated parameters
        st.subheader("📊 Calculated Parameters")
        try:
            tank_weight = calculate_tank_weight_from_thickness(insulation_thickness_mm)
            st.metric("Tank Weight", f"{tank_weight:.2f} kg")
            
            # Calculate heat leak (simplified)
            # Use tank radius from session state if AI recommendations were applied
            current_tank_radius = st.session_state.get('tank_radius_m', 0.3)
            tank = LH2Tank(tank_radius=current_tank_radius, tank_length=1.5)
            tank.mli_thickness = insulation_thickness_mm / 1000.0
            heat_leak = tank.calculate_total_heat_leak()
            boil_off_rate = tank.calculate_boil_off_rate()
            st.metric("Heat Leak", f"{heat_leak:.2f} W")
            st.metric("Boil-Off Rate", f"{boil_off_rate:.4f} kg/h")
        except Exception as e:
            st.error(f"⚠️ Error calculating tank parameters: {str(e)}")
            st.info("Please check your input values and try again.")
            tank_weight = 0.5  # Default fallback
            heat_leak = 0.0
            boil_off_rate = 0.0
        
        # Download Report Section
        st.markdown("---")
        st.subheader("📥 Export Report")
        
        # Collect current simulation data for export
        try:
            current_range_export = calculate_range_with_tank_weight(
                tank_weight, fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                lift_to_drag=15.0, total_efficiency=0.5, cruise_velocity=cruise_velocity
            )
        except ValueError as e:
            st.error(f"⚠️ Cannot calculate range: {str(e)}")
            st.info("💡 Tip: Ensure fuel mass > 0, empty mass > 0, and cruise velocity > 0")
            current_range_export = 0.0
        
        # Prepare data for CSV export
        report_data = {
            'Parameter': [
                'Insulation Thickness (mm)',
                'Tank Weight (kg)',
                'Fuel Cell Power (kW)',
                'Cruise Velocity (m/s)',
                'Payload Mass (kg)',
                'Fuel Mass (kg)',
                'Mission Range (km)',
                'Heat Leak (W)',
                'Boil-Off Rate (kg/h)',
                'Lift-to-Drag Ratio',
                'System Efficiency',
                'Timestamp'
            ],
            'Value': [
                f"{insulation_thickness_mm:.1f}",
                f"{tank_weight:.2f}",
                f"{fuel_cell_power_kw:.1f}",
                f"{cruise_velocity:.1f}",
                "5.0",
                "2.0",
                f"{current_range_export:.2f}",
                f"{heat_leak:.2f}",
                f"{boil_off_rate:.4f}",
                "15.0",
                "0.5",
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        }
        
        df_report = pd.DataFrame(report_data)
        
        # Convert DataFrame to CSV
        csv_data = df_report.to_csv(index=False)
        
        # Download button
        st.download_button(
            label="📥 Download Simulation Report (CSV)",
            data=csv_data,
            file_name=f"hydrogen_uav_simulation_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download current simulation parameters and results as CSV file"
        )
        
        st.caption("Report includes all current system parameters and calculated results")
        
        # AI Design Optimizer Section
        st.markdown("---")
        st.subheader("🤖 AI Design Optimizer")
        st.markdown("""
        **Powered by Aether-Agent (Space LLM)**
        
        Get intelligent design optimization recommendations and automatically apply them to improve performance.
        """)
        
        # LLM API Configuration
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["OpenAI", "Google Generative AI"],
            index=0,
            help="Select the LLM provider for Aether-Agent analysis"
        )
        
        # API Key input (masked for security)
        api_key = st.text_input(
            "LLM API Key",
            type="password",
            value=st.session_state.get('llm_api_key', ''),
            help="Enter your API key for the selected LLM provider. Your key is stored securely in session state.",
            key="llm_api_key_input"
        )
        
        # Validate API key format (basic validation)
        if api_key:
            # Basic validation: check if it's not empty and has reasonable length
            if len(api_key.strip()) < 10:
                st.warning("⚠️ API key seems too short. Please verify your API key.")
            elif len(api_key) > 200:
                st.warning("⚠️ API key seems too long. Please verify your API key.")
            else:
                st.session_state['llm_api_key'] = api_key.strip()
        elif 'llm_api_key' not in st.session_state:
            st.session_state['llm_api_key'] = ''
        
        # Show API key status
        if st.session_state.get('llm_api_key'):
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Please enter an API key to use the AI Design Optimizer")
        
        # Add tank radius slider (if not already present)
        if 'tank_radius_m' not in st.session_state:
            st.session_state['tank_radius_m'] = 0.3
        
        # System prompt for Aether-Agent
        SYSTEM_PROMPT = """You are Aether-Agent, an expert Aerospace Propulsion Engineer. Analyze the following UAV simulation data and provide three specific, mathematically-grounded recommendations to improve mission range.

Your response must include:
1. A detailed engineering analysis of the current system configuration
2. Three specific, actionable recommendations with mathematical justification
3. A JSON object with suggested parameter values in this exact format:
{
  "insulation_thickness": <float in mm>,
  "tank_radius": <float in meters>,
  "cruise_velocity": <float in m/s>
}

Focus on:
- Thermal management and boil-off rate optimization
- Propulsion efficiency improvements
- Weight reduction strategies
- Range maximization through parameter optimization

Provide recommendations that are physically realistic and based on aerospace engineering principles."""
        
        # Real API call function for Aether-Agent
        def call_aether_agent(simulation_data_str, provider):
            """
            Real API call function for Aether-Agent Space LLM using OpenAI or Google Generative AI.
            
            Returns both text recommendations and structured JSON with suggested parameter values.
            
            Args:
                simulation_data_str (str): Formatted string containing simulation results
                provider (str): LLM provider name ("OpenAI" or "Google Generative AI")
                
            Returns:
                dict: Dictionary containing:
                    - 'recommendation': str - Text engineering recommendation
                    - 'suggestions': dict - Structured JSON with suggested parameter values:
                        - 'insulation_thickness': float (mm)
                        - 'tank_radius': float (m)
                        - 'cruise_velocity': float (m/s)
            """
            api_key = st.session_state.get('llm_api_key', '')
            
            if not api_key:
                # Fallback to mock if no API key provided
                st.warning("⚠️ No API key provided. Using mock recommendations. Please enter an API key for real AI analysis.")
                return call_aether_agent_mock(simulation_data_str)
            
            try:
                if provider == "OpenAI":
                    return call_openai_api(simulation_data_str, api_key)
                elif provider == "Google Generative AI":
                    return call_google_generative_ai_api(simulation_data_str, api_key)
                else:
                    st.error(f"Unknown LLM provider: {provider}")
                    return call_aether_agent_mock(simulation_data_str)
            except Exception as e:
                st.error(f"❌ API Error: {str(e)}")
                st.info("Falling back to mock recommendations. Please check your API key and connection.")
                return call_aether_agent_mock(simulation_data_str)
        
        def call_openai_api(simulation_data_str, api_key):
            """Call OpenAI API for Aether-Agent analysis."""
            try:
                from openai import OpenAI
                
                client = OpenAI(api_key=api_key)
                
                user_prompt = f"""{simulation_data_str}

Please provide your analysis and recommendations in the following format:

**Engineering Analysis:**
[Your detailed analysis here]

**Recommendations:**
1. [First recommendation with mathematical justification]
2. [Second recommendation with mathematical justification]
3. [Third recommendation with mathematical justification]

**Suggested Parameters (JSON):**
{{
  "insulation_thickness": <value>,
  "tank_radius": <value>,
  "cruise_velocity": <value>
}}"""
                
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                
                response_text = response.choices[0].message.content
                
                # Extract JSON from response
                suggested_values = extract_json_from_response(response_text)
                
                # Format recommendation
                full_recommendation = f"""**Aether-Agent Engineering Analysis**

{response_text}

---
*Analysis generated by Aether-Agent Space LLM (OpenAI GPT-4) | Real-time AI recommendations*"""
                
                return {
                    'recommendation': full_recommendation,
                    'suggestions': suggested_values
                }
                
            except Exception as e:
                raise Exception(f"OpenAI API error: {str(e)}")
        
        def call_google_generative_ai_api(simulation_data_str, api_key):
            """Call Google Generative AI API for Aether-Agent analysis."""
            try:
                import google.generativeai as genai
                
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                
                user_prompt = f"""{simulation_data_str}

Please provide your analysis and recommendations in the following format:

**Engineering Analysis:**
[Your detailed analysis here]

**Recommendations:**
1. [First recommendation with mathematical justification]
2. [Second recommendation with mathematical justification]
3. [Third recommendation with mathematical justification]

**Suggested Parameters (JSON):**
{{
  "insulation_thickness": <value>,
  "tank_radius": <value>,
  "cruise_velocity": <value>
}}"""
                
                response = model.generate_content(
                    f"{SYSTEM_PROMPT}\n\n{user_prompt}",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=1500
                    )
                )
                
                response_text = response.text
                
                # Extract JSON from response
                suggested_values = extract_json_from_response(response_text)
                
                # Format recommendation
                full_recommendation = f"""**Aether-Agent Engineering Analysis**

{response_text}

---
*Analysis generated by Aether-Agent Space LLM (Google Gemini Pro) | Real-time AI recommendations*"""
                
                return {
                    'recommendation': full_recommendation,
                    'suggestions': suggested_values
                }
                
            except Exception as e:
                raise Exception(f"Google Generative AI API error: {str(e)}")
        
        def extract_json_from_response(response_text):
            """Extract JSON object from LLM response text."""
            # Try to find JSON in the response
            json_match = re.search(r'\{[^{}]*"insulation_thickness"[^{}]*\}', response_text, re.DOTALL)
            
            if json_match:
                try:
                    json_str = json_match.group(0)
                    suggested_values = json.loads(json_str)
                    # Validate and set defaults if missing
                    if 'insulation_thickness' not in suggested_values:
                        suggested_values['insulation_thickness'] = 50.0
                    if 'tank_radius' not in suggested_values:
                        suggested_values['tank_radius'] = 0.3
                    if 'cruise_velocity' not in suggested_values:
                        suggested_values['cruise_velocity'] = 30.0
                    return suggested_values
                except json.JSONDecodeError:
                    pass
            
            # Fallback: parse values from text
            return parse_values_from_text(response_text)
        
        def parse_values_from_text(text):
            """Parse parameter values from text if JSON extraction fails."""
            suggested_values = {
                'insulation_thickness': 50.0,
                'tank_radius': 0.3,
                'cruise_velocity': 30.0
            }
            
            # Try to extract insulation thickness
            ins_match = re.search(r'insulation[_\s]*thickness[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
            if ins_match:
                suggested_values['insulation_thickness'] = float(ins_match.group(1))
            
            # Try to extract tank radius
            radius_match = re.search(r'tank[_\s]*radius[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
            if radius_match:
                suggested_values['tank_radius'] = float(radius_match.group(1))
            
            # Try to extract cruise velocity
            vel_match = re.search(r'cruise[_\s]*velocity[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
            if vel_match:
                suggested_values['cruise_velocity'] = float(vel_match.group(1))
            
            return suggested_values
        
        def call_aether_agent_mock(simulation_data_str):
            """Fallback mock function when API is unavailable."""
            # Parse key metrics from the data string
            current_insulation = 50.0
            current_tank_radius = 0.3
            current_cruise_velocity = 30.0
            current_range = 0.0
            current_bor = 0.0
            
            try:
                if "Insulation Thickness:" in simulation_data_str:
                    ins_line = [line for line in simulation_data_str.split('\n') if 'Insulation Thickness:' in line][0]
                    current_insulation = float(ins_line.split(':')[1].split()[0])
                
                if "Tank Radius:" in simulation_data_str:
                    radius_line = [line for line in simulation_data_str.split('\n') if 'Tank Radius:' in line][0]
                    current_tank_radius = float(radius_line.split(':')[1].split()[0])
                
                if "Cruise Velocity:" in simulation_data_str:
                    vel_line = [line for line in simulation_data_str.split('\n') if 'Cruise Velocity:' in line][0]
                    current_cruise_velocity = float(vel_line.split(':')[1].split()[0])
                
                if "Mission Range:" in simulation_data_str:
                    range_line = [line for line in simulation_data_str.split('\n') if 'Mission Range:' in line][0]
                    current_range = float(range_line.split(':')[1].split()[0])
                
                if "Boil-off Rate:" in simulation_data_str:
                    bor_line = [line for line in simulation_data_str.split('\n') if 'Boil-off Rate:' in line][0]
                    current_bor = float(bor_line.split(':')[1].split()[0])
            except:
                pass
            
            # Generate mock recommendations
            recommendations = [
                f"⚠️ **MOCK MODE**: Using rule-based recommendations. Enter an API key for real AI analysis.",
                f"**Current Range**: {current_range:.2f} km",
                f"**Recommendation**: Optimize insulation thickness and cruise velocity for improved performance."
            ]
            
            suggested_values = {
                'insulation_thickness': min(100, current_insulation + 10),
                'tank_radius': current_tank_radius,
                'cruise_velocity': max(15, current_cruise_velocity - 2)
            }
            
            recommendation_text = "\n\n".join(recommendations)
            full_recommendation = f"""**Aether-Agent Engineering Analysis (Mock Mode)**

{recommendation_text}

---
*Mock recommendations - Enter API key for real AI analysis*"""
            
            return {
                'recommendation': full_recommendation,
                'suggestions': suggested_values
            }
        
        # Collect current simulation data
        try:
            current_range_ai = calculate_range_with_tank_weight(
                tank_weight, fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                lift_to_drag=15.0, total_efficiency=0.5, cruise_velocity=cruise_velocity
            )
        except ValueError as e:
            st.error(f"⚠️ Cannot calculate range: {str(e)}")
            st.info("💡 Tip: Ensure fuel mass > 0, empty mass > 0, and cruise velocity > 0")
            current_range_ai = 0.0
        
        # Format simulation data as string
        current_tank_radius_display = st.session_state.get('tank_radius_m', 0.3)
        simulation_data = f"""Hydrogen-Electric UAV Propulsion System Analysis

Current Configuration:
- Insulation Thickness: {insulation_thickness_mm:.1f} mm
- Tank Radius: {current_tank_radius_display:.2f} m
- Tank Weight: {tank_weight:.2f} kg
- Fuel Cell Power: {fuel_cell_power_kw:.1f} kW
- Cruise Velocity: {cruise_velocity:.1f} m/s
- Mission Range: {current_range_ai:.2f} km
- Heat Leak: {heat_leak:.2f} W
- Boil-off Rate: {boil_off_rate:.4f} kg/h
- System Efficiency: 50.0%
- Payload Mass: 5.0 kg
- Fuel Mass: 2.0 kg
- Lift-to-Drag Ratio: 15.0

Request: Provide engineering recommendations and optimized parameter values for system optimization."""
        
        # Consult AI button
        if st.button("🔮 Consult AI", type="primary", use_container_width=True):
            if not st.session_state.get('llm_api_key'):
                st.warning("⚠️ Please enter an API key in the field above to use the AI Design Optimizer.")
            else:
                with st.spinner(f"🤖 Consulting Aether-Agent Space LLM ({llm_provider})..."):
                    try:
                        result = call_aether_agent(simulation_data, llm_provider)
                        st.session_state['aether_recommendation'] = result['recommendation']
                        st.session_state['aether_suggestions'] = result['suggestions']
                        st.session_state['aether_data'] = simulation_data
                        # Store before state when consulting AI
                        st.session_state['before_state'] = {
                            'insulation_thickness': insulation_thickness_mm,
                            'tank_radius': st.session_state.get('tank_radius_m', 0.3),
                            'cruise_velocity': cruise_velocity,
                            'range': current_range_ai,
                            'boil_off_rate': boil_off_rate,
                            'heat_leak': heat_leak,
                            'tank_weight': tank_weight
                        }
                    except Exception as e:
                        st.error(f"❌ Error calling AI API: {str(e)}")
                        st.info("Please check your API key and try again.")
        
        # Display recommendation if available
        if 'aether_recommendation' in st.session_state:
            st.markdown("---")
            # Style Aether-Agent output in a prominent info box
            st.info("""
            **🤖 Aether-Agent Engineering Analysis**
            
            """ + st.session_state['aether_recommendation'])
            
            # Display suggested values
            if 'aether_suggestions' in st.session_state:
                st.markdown("### 🎯 AI Suggested Parameter Values")
                suggestions = st.session_state['aether_suggestions']
                col_sug1, col_sug2, col_sug3 = st.columns(3)
                with col_sug1:
                    st.metric("Insulation Thickness", f"{suggestions['insulation_thickness']:.1f} mm",
                             delta=f"{suggestions['insulation_thickness'] - insulation_thickness_mm:+.1f} mm")
                with col_sug2:
                    st.metric("Tank Radius", f"{suggestions['tank_radius']:.2f} m",
                             delta=f"{suggestions['tank_radius'] - st.session_state.get('tank_radius_m', 0.3):+.2f} m")
                with col_sug3:
                    st.metric("Cruise Velocity", f"{suggestions['cruise_velocity']:.1f} m/s",
                             delta=f"{suggestions['cruise_velocity'] - cruise_velocity:+.1f} m/s")
            
            # Apply AI Recommendations button
            if st.button("✅ Apply AI Recommendations", type="secondary", use_container_width=True):
                if 'aether_suggestions' in st.session_state:
                    suggestions = st.session_state['aether_suggestions']
                    # Store before state for comparison
                    st.session_state['before_state'] = {
                        'insulation_thickness': insulation_thickness_mm,
                        'tank_radius': st.session_state.get('tank_radius_m', 0.3),
                        'cruise_velocity': cruise_velocity,
                        'range': current_range_ai,
                        'boil_off_rate': boil_off_rate,
                        'heat_leak': heat_leak,
                        'tank_weight': tank_weight
                    }
                    # Update session state to trigger re-run with new values
                    st.session_state['apply_ai_recommendations'] = True
                    st.session_state['ai_insulation'] = suggestions['insulation_thickness']
                    st.session_state['ai_tank_radius'] = suggestions['tank_radius']
                    st.session_state['ai_cruise_velocity'] = suggestions['cruise_velocity']
                    st.session_state['ai_applied_once'] = True
                    st.rerun()
            
            # Option to view raw data
            with st.expander("📊 View Simulation Data Sent to AI"):
                st.code(st.session_state['aether_data'], language='text')
        
        # Display Before vs After comparison if recommendations were applied
        if 'before_state' in st.session_state and st.session_state.get('ai_applied_once', False):
            st.markdown("---")
            st.markdown("### 📊 Before vs. After Comparison")
            
            # Recalculate "after" state with current values (use AI-applied values)
            current_insulation_after = st.session_state.get('ai_insulation', insulation_thickness_mm)
            current_tank_radius_after = st.session_state.get('ai_tank_radius', st.session_state.get('tank_radius_m', 0.3))
            current_velocity_after = st.session_state.get('ai_cruise_velocity', cruise_velocity)
            
            tank_after = LH2Tank(tank_radius=current_tank_radius_after, tank_length=1.5)
            tank_after.mli_thickness = current_insulation_after / 1000.0
            heat_leak_after = tank_after.calculate_total_heat_leak()
            boil_off_rate_after = tank_after.calculate_boil_off_rate()
            tank_weight_after = calculate_tank_weight_from_thickness(current_insulation_after)
            range_after = calculate_range_with_tank_weight(
                tank_weight_after, fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                lift_to_drag=15.0, total_efficiency=0.5, cruise_velocity=current_velocity_after
            )
            
            before = st.session_state['before_state']
            after_state = {
                'insulation_thickness': current_insulation_after,
                'tank_radius': current_tank_radius_after,
                'cruise_velocity': current_velocity_after,
                'range': range_after,
                'boil_off_rate': boil_off_rate_after,
                'heat_leak': heat_leak_after,
                'tank_weight': tank_weight_after
            }
            
            # Create comparison table
            comparison_data = {
                'Parameter': [
                    'Insulation Thickness (mm)',
                    'Tank Radius (m)',
                    'Cruise Velocity (m/s)',
                    'Mission Range (km)',
                    'Boil-off Rate (kg/h)',
                    'Heat Leak (W)',
                    'Tank Weight (kg)'
                ],
                'Before': [
                    f"{before['insulation_thickness']:.1f}",
                    f"{before['tank_radius']:.2f}",
                    f"{before['cruise_velocity']:.1f}",
                    f"{before['range']:.2f}",
                    f"{before['boil_off_rate']:.4f}",
                    f"{before['heat_leak']:.2f}",
                    f"{before['tank_weight']:.2f}"
                ],
                'After': [
                    f"{after_state['insulation_thickness']:.1f}",
                    f"{after_state['tank_radius']:.2f}",
                    f"{after_state['cruise_velocity']:.1f}",
                    f"{after_state['range']:.2f}",
                    f"{after_state['boil_off_rate']:.4f}",
                    f"{after_state['heat_leak']:.2f}",
                    f"{after_state['tank_weight']:.2f}"
                ],
                'Change': [
                    f"{after_state['insulation_thickness'] - before['insulation_thickness']:+.1f}",
                    f"{after_state['tank_radius'] - before['tank_radius']:+.2f}",
                    f"{after_state['cruise_velocity'] - before['cruise_velocity']:+.1f}",
                    f"{after_state['range'] - before['range']:+.2f}",
                    f"{after_state['boil_off_rate'] - before['boil_off_rate']:+.4f}",
                    f"{after_state['heat_leak'] - before['heat_leak']:+.2f}",
                    f"{after_state['tank_weight'] - before['tank_weight']:+.2f}"
                ],
                'Improvement': [
                    '✅' if abs(after_state['insulation_thickness'] - before['insulation_thickness']) > 0 else '',
                    '✅' if abs(after_state['tank_radius'] - before['tank_radius']) > 0 else '',
                    '✅' if abs(after_state['cruise_velocity'] - before['cruise_velocity']) > 0 else '',
                    '✅' if after_state['range'] > before['range'] else '⚠️' if after_state['range'] < before['range'] else '',
                    '✅' if after_state['boil_off_rate'] < before['boil_off_rate'] else '⚠️' if after_state['boil_off_rate'] > before['boil_off_rate'] else '',
                    '✅' if after_state['heat_leak'] < before['heat_leak'] else '⚠️' if after_state['heat_leak'] > before['heat_leak'] else '',
                    '✅' if after_state['tank_weight'] < before['tank_weight'] else '⚠️' if after_state['tank_weight'] > before['tank_weight'] else ''
                ]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
            
            # Summary metrics
            range_improvement = ((after_state['range'] - before['range']) / before['range']) * 100
            bor_improvement = ((before['boil_off_rate'] - after_state['boil_off_rate']) / before['boil_off_rate']) * 100 if before['boil_off_rate'] > 0 else 0
            
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                st.metric("Range Change", f"{range_improvement:+.1f}%",
                         delta=f"{after_state['range'] - before['range']:+.2f} km")
            with col_sum2:
                st.metric("Boil-off Reduction", f"{bor_improvement:+.1f}%",
                         delta=f"{after_state['boil_off_rate'] - before['boil_off_rate']:+.4f} kg/h")
            with col_sum3:
                if range_improvement > 0 and bor_improvement > 0:
                    st.success("✅ **AI Optimization Successful!**")
                elif range_improvement > 0:
                    st.info("ℹ️ Range improved, but boil-off increased")
                else:
                    st.warning("⚠️ Trade-off detected - review recommendations")
        
        # Info about Aether-Agent integration
        with st.expander("ℹ️ About Aether-Agent Integration"):
            st.markdown("""
            **Aether-Agent Space LLM Integration**
            
            This feature connects your propulsion simulation tool with the Aether-Agent 
            Space Large Language Model for intelligent engineering analysis.
            
            **Current Status**: Placeholder function simulating API calls
            
            **To Enable Full Integration**:
            1. Replace `call_aether_agent()` with actual Aether-Agent API endpoint
            2. Configure API authentication (API key, OAuth, etc.)
            3. Update request/response format to match Aether-Agent API specification
            4. Add error handling for network failures and API rate limits
            
            **Example API Call Structure**:
            ```python
            import requests
            
            def call_aether_agent(data):
                response = requests.post(
                    'https://api.aether-agent.space/v1/analyze',
                    headers={'Authorization': f'Bearer {API_KEY}'},
                    json={'simulation_data': data}
                )
                return response.json()['recommendation']
            ```
            
            **Benefits**:
            - Real-time engineering insights
            - Context-aware recommendations
            - Learning from historical data
            - Multi-objective optimization suggestions
            """)
    
    # Main content area with tabs
    with st.container():
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Performance Analysis", "🎯 Break-Even Analysis", 
                                         "📊 Reliability & Risk", "🛡️ Safety & Compliance", 
                                         "🏆 Industry Benchmarks", "📊 Trade-Space Analysis"])
    
    with tab1:
        # Dashboard-style Performance Analysis
        st.header("📊 Performance Analysis Dashboard")
        st.markdown("---")
        
        # Calculate all data upfront for efficiency
        fuel_cell = PEMFuelCell(E_ocv=1.2, temperature=353.15)
        max_current_density = fuel_cell_power_kw * 200
        current_densities = np.linspace(0, max_current_density, 200)
        cell_voltages = fuel_cell.calculate_cell_voltage(current_densities)
        power_densities = fuel_cell.calculate_power_density(current_densities)
        
        # Calculate sensitivity analysis data
        tank_weights = np.linspace(0.3, 1.5, 50)
        # Calculate ranges with error handling
        ranges_h2 = []
        for tw in tank_weights:
            try:
                range_val = calculate_range_with_tank_weight(
                    tw, fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                    lift_to_drag=15.0, total_efficiency=0.5, cruise_velocity=cruise_velocity
                )
                ranges_h2.append(range_val)
            except ValueError:
                ranges_h2.append(0.0)  # Default to 0 if calculation fails
        h2_energy = 2.0 * 120e6
        liion_energy_density_jkg = 250 * 3600
        equivalent_battery_mass = h2_energy / liion_energy_density_jkg
        base_empty_mass = 10.0
        ranges_liion = [calculate_liion_range(equivalent_battery_mass, 
                                               payload_mass=5.0,
                                               empty_mass=base_empty_mass,
                                               lift_to_drag=15.0,
                                               total_efficiency=0.5) 
                        for _ in tank_weights]
        current_tank_weight = calculate_tank_weight_from_thickness(insulation_thickness_mm)
        try:
            current_range = calculate_range_with_tank_weight(
                current_tank_weight, fuel_mass=2.0, payload_mass=5.0,
                empty_mass=10.0, lift_to_drag=15.0, total_efficiency=0.5,
                cruise_velocity=cruise_velocity
            )
        except ValueError as e:
            st.error(f"⚠️ Cannot calculate range: {str(e)}")
            st.info("💡 Tip: Ensure fuel mass > 0, empty mass > 0, and cruise velocity > 0")
            current_range = 0.0
        
        # Calculate energy balance data
        fuel_mass_energy = 2.0
        hydrogen_energy_density = 120e6
        total_energy_j = fuel_mass_energy * hydrogen_energy_density
        total_energy_mj = total_energy_j / 1e6
        fuel_cell_efficiency = 0.55
        fuel_cell_output_energy = total_energy_j * fuel_cell_efficiency
        fuel_cell_heat_loss = total_energy_j * (1 - fuel_cell_efficiency)
        motor_inverter_efficiency = 0.88
        motor_output_energy = fuel_cell_output_energy * motor_inverter_efficiency
        motor_inverter_loss = fuel_cell_output_energy * (1 - motor_inverter_efficiency)
        try:
            current_range_energy = calculate_range_with_tank_weight(
                tank_weight, fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                lift_to_drag=15.0, total_efficiency=0.5, cruise_velocity=cruise_velocity
            )
        except ValueError:
            current_range_energy = 100.0  # Default fallback for energy calculations
        total_system_efficiency = 0.5
        useful_propulsive_energy = total_energy_j * total_system_efficiency
        propulsive_losses = fuel_cell_output_energy - useful_propulsive_energy
        latent_heat_vaporization = 448e3
        mission_duration_hours = current_range_energy / (cruise_velocity * 3.6) if cruise_velocity > 0 else 2.0
        boil_off_mass = boil_off_rate * mission_duration_hours
        bog_energy_loss = boil_off_mass * latent_heat_vaporization
        useful_energy = useful_propulsive_energy - bog_energy_loss
        fuel_cell_heat_loss_mj = fuel_cell_heat_loss / 1e6
        motor_inverter_loss_mj = motor_inverter_loss / 1e6
        propulsive_losses_mj = propulsive_losses / 1e6
        bog_energy_loss_mj = bog_energy_loss / 1e6
        useful_energy_mj = useful_energy / 1e6
        
        # Dashboard Grid Layout: 2x2 for main visualizations
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        # Row 1, Column 1: Polarization Curve
        with row1_col1:
            with st.expander("📈 Polarization Curve Analysis", expanded=True):
                st.markdown("""
                **Fuel Cell Performance Visualization**
                
                The polarization curve shows how cell voltage decreases with increasing current density.
                """)
                
                # Create smaller default plot
                fig1, ax1 = plt.subplots(figsize=(6, 4))
        
                # Plot voltage vs current density
                ax1.plot(current_densities, cell_voltages, 'b-', linewidth=2, label='Cell Voltage')
                ax1.set_xlabel('Current Density (A/m²)', fontsize=10, fontweight='bold')
                ax1.set_ylabel('Cell Voltage (V)', fontsize=10, fontweight='bold')
                ax1.set_title('PEM Fuel Cell Polarization Curve', 
                             fontsize=12, fontweight='bold', pad=10)
                ax1.grid(True, alpha=0.3, linestyle='--')
                
                # Add efficiency zones
                efficiency_threshold = 0.6
                threshold_voltage = fuel_cell.E_ocv * efficiency_threshold
                ax1.axhline(y=threshold_voltage, color='r', linestyle='--', alpha=0.5, 
                           label=f'60% Threshold ({threshold_voltage:.2f}V)')
                
                # Highlight concentration loss region
                high_current_idx = np.where(current_densities > max_current_density * 0.7)[0]
                if len(high_current_idx) > 0:
                    ax1.fill_between(current_densities[high_current_idx], 
                                    cell_voltages[high_current_idx], 
                                    threshold_voltage, alpha=0.2, color='red',
                                    label='Concentration Loss')
                
                ax1.set_xlim([0, max_current_density])
                ax1.set_ylim([0, fuel_cell.E_ocv * 1.1])
                ax1.legend(fontsize=9, loc='upper right')
                
                plt.tight_layout()
                st.pyplot(fig1, use_container_width=True)
                
                # Display key metrics compactly
                max_power_idx = np.argmax(power_densities)
                max_power_current = current_densities[max_power_idx]
                max_power_voltage = cell_voltages[max_power_idx]
                max_power_density = power_densities[max_power_idx]
                
                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    st.metric("Max Power Density", f"{max_power_density:.1f} W/m²")
                    st.metric("Optimal Current", f"{max_power_current:.1f} A/m²")
                with col_met2:
                    st.metric("Voltage @ Max Power", f"{max_power_voltage:.3f} V")
                    st.metric("Stack Power", f"{fuel_cell_power_kw:.1f} kW")
        
        # Row 1, Column 2: Sensitivity Analysis
        with row1_col2:
            with st.expander("🔬 Sensitivity Analysis", expanded=True):
                st.markdown("""
                **Range vs. Tank Weight Trade-Study**
                
                Trade-off: thicker insulation reduces boil-off but increases weight.
                """)
                
                # Create smaller sensitivity plot
                fig2, ax2 = plt.subplots(figsize=(6, 4))
        
                # Plot hydrogen system
                ax2.plot(tank_weights, ranges_h2, 'b-', linewidth=2, 
                        label='H₂ System (2 kg)', marker='o', markersize=3)
                
                # Plot Li-ion comparison
                ax2.plot(tank_weights, ranges_liion, 'r--', linewidth=2, 
                        label=f'Li-ion ({equivalent_battery_mass:.1f} kg)', 
                        marker='s', markersize=3)
                
                # Highlight current design point
                ax2.plot(current_tank_weight, current_range, 'go', markersize=10, 
                        label='Current Design', zorder=5)
                
                ax2.set_xlabel('Tank Weight (kg)', fontsize=10, fontweight='bold')
                ax2.set_ylabel('Range (km)', fontsize=10, fontweight='bold')
                ax2.set_title('Range vs. Tank Weight', 
                             fontsize=12, fontweight='bold', pad=10)
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.legend(fontsize=9, loc='best')
                
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                
                # Display key metrics compactly
                h2_specific_energy = 120e6 / 1000 / 3600
                liion_specific_energy = 250
                advantage_ratio = h2_specific_energy / liion_specific_energy
                
                col_sens1, col_sens2 = st.columns(2)
                with col_sens1:
                    st.metric("Current Range", f"{current_range:.2f} km")
                    st.metric("Tank Weight", f"{current_tank_weight:.2f} kg")
                with col_sens2:
                    st.metric("H₂ Energy Density", f"{h2_specific_energy:.0f} Wh/kg")
                    st.metric("H₂ Advantage", f"{advantage_ratio:.1f}x")
        
        # Row 2, Column 1: Energy Balance Bar Chart
        with row2_col1:
            with st.expander("⚡ Energy Balance Analysis", expanded=True):
                st.markdown("""
                **Energy Distribution Visualization**
                
                Shows how hydrogen fuel energy flows through the propulsion system.
                """)
        
                # Create Energy Balance Bar Chart
                energy_categories = [
                    'Total Energy\n(H₂ Fuel)',
                    'Fuel Cell\nHeat Loss',
                    'Motor/Inverter\nLosses',
                    'Aerodynamic\nDrag Losses',
                    'Boil-off Gas\n(BOG) Loss',
                    'Useful\nPropulsive Energy'
                ]
                
                energy_values = [
                    total_energy_mj,
                    -fuel_cell_heat_loss_mj,
                    -motor_inverter_loss_mj,
                    -propulsive_losses_mj,
                    -bog_energy_loss_mj,
                    useful_energy_mj
                ]
                
                colors = [
                    '#1E88E5',  # Bright blue for total energy
                    '#D32F2F',  # Deep red for fuel cell losses
                    '#F57C00',  # Deep orange for motor losses
                    '#FBC02D',  # Amber/yellow for drag losses
                    '#7B1FA2',  # Rich purple for BOG losses
                    '#388E3C'   # Forest green for useful energy
                ]
                
                # Create Plotly bar chart with reduced height
                fig_energy = go.Figure()
                
                loss_categories = energy_categories[1:5]
                loss_values = energy_values[1:5]
                loss_colors = colors[1:5]
                
                fig_energy.add_trace(go.Bar(
                    x=loss_categories,
                    y=[abs(v) for v in loss_values],
                    marker=dict(
                        color=loss_colors,
                        line=dict(color='#333', width=1),
                        opacity=0.85
                    ),
                    name='Energy Losses',
                    text=[f'{abs(v):.2f} MJ' for v in loss_values],
                    textposition='outside',
                    textfont=dict(size=9, color='#333', family='Arial, sans-serif'),
                    hovertemplate='<b>%{x}</b><br>Energy Loss: %{y:.2f} MJ<extra></extra>'
                ))
                
                fig_energy.add_trace(go.Bar(
                    x=[energy_categories[5]],
                    y=[energy_values[5]],
                    marker=dict(
                        color=colors[5],
                        line=dict(color='#2E7D32', width=1.5),
                        opacity=0.9
                    ),
                    name='Useful Energy',
                    text=[f'{energy_values[5]:.2f} MJ'],
                    textposition='outside',
                    textfont=dict(size=10, color='#2E7D32', family='Arial, sans-serif', weight='bold'),
                    hovertemplate='<b>%{x}</b><br>Useful Energy: %{y:.2f} MJ<extra></extra>'
                ))
                
                overall_efficiency = (useful_energy_mj / total_energy_mj) * 100
                fig_energy.update_layout(
                    title={
                        'text': f'Energy Balance<br><sub>Total: {total_energy_mj:.2f} MJ | Efficiency: {overall_efficiency:.1f}%</sub>',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 13, 'family': 'Arial, sans-serif', 'color': '#1a1a1a'}
                    },
                    xaxis=dict(
                        title=dict(text='Energy Category', font=dict(size=12, family='Arial, sans-serif', color='#000000', weight='bold')),
                        tickfont=dict(size=10, family='Arial, sans-serif', color='#000000'),
                        gridcolor='#d0d0d0',
                        gridwidth=1,
                        showgrid=True,
                        linecolor='#000000',
                        linewidth=2,
                        mirror=True,
                        showline=True
                    ),
                    yaxis=dict(
                        title=dict(text='Energy (MJ)', font=dict(size=12, family='Arial, sans-serif', color='#000000', weight='bold')),
                        tickfont=dict(size=10, family='Arial, sans-serif', color='#000000'),
                        gridcolor='#d0d0d0',
                        gridwidth=1,
                        showgrid=True,
                        linecolor='#000000',
                        linewidth=2,
                        mirror=True,
                        showline=True
                    ),
                    barmode='group',
                    height=350,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=10, family='Arial, sans-serif'),
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='#ccc',
                        borderwidth=1
                    ),
                    plot_bgcolor='#fafafa',
                    paper_bgcolor='white',
                    font=dict(size=10, family='Arial, sans-serif'),
                    margin=dict(l=50, r=20, t=80, b=50)
                )
                
                st.plotly_chart(fig_energy, use_container_width=True)
                
                # Compact efficiency metrics
                fuel_cell_eff = (fuel_cell_output_energy / total_energy_j) * 100
                motor_eff = (motor_output_energy / fuel_cell_output_energy) * 100
                propulsive_eff = (useful_propulsive_energy / motor_output_energy) * 100
                bog_percentage = (bog_energy_loss_mj / total_energy_mj) * 100
                
                col_eff1, col_eff2 = st.columns(2)
                with col_eff1:
                    st.metric("System Efficiency", f"{overall_efficiency:.1f}%")
                    st.metric("Fuel Cell Eff.", f"{fuel_cell_eff:.1f}%")
                with col_eff2:
                    st.metric("Motor/Inverter Eff.", f"{motor_eff:.1f}%")
                    st.metric("BOG Loss", f"{bog_percentage:.2f}%")
        
        # Row 2, Column 2: Sankey Diagram
        with row2_col2:
            with st.expander("🔄 Energy Flow Diagram (Sankey)", expanded=True):
                st.markdown("""
                **Energy Flow Visualization**
                
                Sankey diagram showing energy conversion pathways.
                """)
        
                # Sankey diagram data
                labels = [
                    f'H₂ Fuel\n({total_energy_mj:.0f} MJ)',
                    'Fuel Cell\nOutput',
                    'Motor/Inverter\nOutput',
                    'Propulsive\nEnergy',
                    'Fuel Cell\nHeat Loss',
                    'Motor/Inverter\nLosses',
                    'Drag Losses',
                    'BOG Losses',
                    'Useful\nEnergy'
                ]
                
                source = [0, 0, 1, 1, 2, 2, 3, 3]
                target = [1, 4, 2, 5, 3, 6, 8, 7]
                value = [
                    fuel_cell_output_energy / 1e6,
                    fuel_cell_heat_loss_mj,
                    motor_output_energy / 1e6,
                    motor_inverter_loss_mj,
                    useful_propulsive_energy / 1e6,
                    propulsive_losses_mj,
                    useful_energy_mj,
                    bog_energy_loss_mj
                ]
                
                link_colors = [
                    '#2E86AB', '#F24236', '#2E86AB', '#FF6B35',
                    '#2E86AB', '#FFA500', '#06A77D', '#9B59B6'
                ]
                
                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=10,
                        thickness=15,
                        line=dict(color="black", width=0.5),
                        label=labels,
                        color=['#2E86AB', '#4A90E2', '#6BA3D8', '#8BB6CE', '#F24236', '#FF6B35', '#FFA500', '#9B59B6', '#06A77D']
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                        color=link_colors
                    )
                )])
                
                fig_sankey.update_layout(
                    title={
                        'text': 'Energy Flow: H₂ Fuel → Propulsive Energy',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 13}
                    },
                    font_size=10,
                    height=400
                )
                
                st.plotly_chart(fig_sankey, use_container_width=True)
        
        # Bottom section: Compact summary and insights
        st.markdown("---")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            with st.expander("📊 Energy Breakdown Table", expanded=False):
                energy_breakdown = pd.DataFrame({
                    'Category': [
                        'Total Energy (H₂)',
                        'Fuel Cell Heat Loss',
                        'Motor/Inverter Losses',
                        'Drag Losses',
                        'BOG Loss',
                        'Useful Energy'
                    ],
                    'Energy (MJ)': [
                        f"{total_energy_mj:.2f}",
                        f"{fuel_cell_heat_loss_mj:.2f}",
                        f"{motor_inverter_loss_mj:.2f}",
                        f"{propulsive_losses_mj:.2f}",
                        f"{bog_energy_loss_mj:.2f}",
                        f"{useful_energy_mj:.2f}"
                    ],
                    '%': [
                        "100.0%",
                        f"{(fuel_cell_heat_loss_mj / total_energy_mj) * 100:.1f}%",
                        f"{(motor_inverter_loss_mj / total_energy_mj) * 100:.1f}%",
                        f"{(propulsive_losses_mj / total_energy_mj) * 100:.1f}%",
                        f"{(bog_energy_loss_mj / total_energy_mj) * 100:.1f}%",
                        f"{(useful_energy_mj / total_energy_mj) * 100:.1f}%"
                    ]
                })
                st.dataframe(energy_breakdown, use_container_width=True, hide_index=True)
        
        with summary_col2:
            with st.expander("⚖️ System Comparison", expanded=False):
                try:
                    h2_range = calculate_range_with_tank_weight(
                        calculate_tank_weight_from_thickness(insulation_thickness_mm),
                        fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                        lift_to_drag=15.0, total_efficiency=0.5, cruise_velocity=cruise_velocity
                    )
                except ValueError:
                    h2_range = 0.0  # Default fallback
                liion_range = calculate_liion_range(
                    equivalent_battery_mass, payload_mass=5.0, 
                    empty_mass=10.0, lift_to_drag=15.0, total_efficiency=0.5
                )
                range_advantage = ((h2_range - liion_range) / liion_range) * 100
                weight_savings = equivalent_battery_mass - (2.0 + current_tank_weight)
                
                st.metric("H₂ Range", f"{h2_range:.2f} km")
                st.metric("Li-ion Range", f"{liion_range:.2f} km")
                st.metric("Range Advantage", f"{range_advantage:+.1f}%")
                st.metric("Weight Savings", f"{weight_savings:+.2f} kg")
                st.metric("Energy Density Ratio", f"{advantage_ratio:.1f}x")
        
        with summary_col3:
            with st.expander("💡 Key Insights", expanded=False):
                overall_efficiency = (useful_energy_mj / total_energy_mj) * 100
                fuel_cell_eff = (fuel_cell_output_energy / total_energy_j) * 100
                bog_percentage = (bog_energy_loss_mj / total_energy_mj) * 100
                
                if bog_percentage > 2.0:
                    st.warning(f"⚠️ High BOG Loss: {bog_percentage:.2f}%")
                elif bog_percentage > 1.0:
                    st.info(f"ℹ️ Moderate BOG Loss: {bog_percentage:.2f}%")
                else:
                    st.success(f"✅ Low BOG Loss: {bog_percentage:.2f}%")
                
                if fuel_cell_eff < 50:
                    st.warning(f"⚠️ Fuel Cell Eff: {fuel_cell_eff:.1f}% (target: 55-60%)")
                else:
                    st.success(f"✅ Fuel Cell Eff: {fuel_cell_eff:.1f}%")
                
                if overall_efficiency < 40:
                    st.warning(f"⚠️ System Eff: {overall_efficiency:.1f}%")
                else:
                    st.success(f"✅ System Eff: {overall_efficiency:.1f}%")
                
                total_losses = fuel_cell_heat_loss_mj + motor_inverter_loss_mj + propulsive_losses_mj + bog_energy_loss_mj
                st.markdown("**Loss Breakdown:**")
                st.markdown(f"- Fuel Cell: {(fuel_cell_heat_loss_mj / total_losses) * 100:.1f}%")
                st.markdown(f"- Motor/Inverter: {(motor_inverter_loss_mj / total_losses) * 100:.1f}%")
                st.markdown(f"- Drag: {(propulsive_losses_mj / total_losses) * 100:.1f}%")
                st.markdown(f"- BOG: {(bog_energy_loss_mj / total_losses) * 100:.1f}%")
    
    with tab2:
        st.header("🎯 Break-Even Analysis Dashboard")
        st.markdown("---")
        
        # Calculate data upfront
        current_tank_weight = calculate_tank_weight_from_thickness(insulation_thickness_mm)
        break_even_range, break_even_mass = find_break_even_range(
            current_tank_weight,
            payload_mass=5.0,
            empty_mass=10.0,
            lift_to_drag=15.0,
            total_efficiency=0.5,
            range_min=10.0,
            range_max=500.0
        )
        ranges_km = np.linspace(10, 300, 100)
        h2_masses = [calculate_h2_system_mass_for_range(r, current_tank_weight, 
                                                        payload_mass=5.0, empty_mass=10.0) 
                     for r in ranges_km]
        liion_masses = [calculate_liion_system_mass_for_range(r, payload_mass=5.0, 
                                                              empty_mass=10.0) 
                       for r in ranges_km]
        
        # Dashboard Layout: Main plot + metrics
        row1_col1, row1_col2 = st.columns([2, 1])
        
        with row1_col1:
            with st.expander("📊 Break-Even Analysis Plot", expanded=True):
                st.markdown("""
                **Critical Range Analysis**: When does hydrogen become lighter than batteries?
                """)
                
                # Create smaller plot
                fig_break_even, ax_be = plt.subplots(figsize=(8, 5))
        
                # Plot system masses
                ax_be.plot(ranges_km, h2_masses, 'b-', linewidth=2.5, 
                           label='H₂ System', alpha=0.8)
                ax_be.plot(ranges_km, liion_masses, 'r--', linewidth=2.5, 
                           label='Li-ion System', alpha=0.8)
                
                # Highlight break-even point
                if break_even_range is not None:
                    ax_be.plot(break_even_range, break_even_mass, 'go', 
                              markersize=12, label=f'Break-Even ({break_even_range:.1f} km)', 
                              zorder=5, markeredgecolor='black', markeredgewidth=1.5)
                    ax_be.axvline(x=break_even_range, color='green', linestyle=':', 
                                 alpha=0.5, linewidth=1.5)
                    ax_be.axhline(y=break_even_mass, color='green', linestyle=':', 
                                 alpha=0.5, linewidth=1.5)
                
                # Fill advantage zones
                h2_array = np.array(h2_masses)
                liion_array = np.array(liion_masses)
                h2_lighter_mask = h2_array < liion_array
                liion_lighter_mask = liion_array < h2_array
                
                if np.any(h2_lighter_mask):
                    ax_be.fill_between(ranges_km[h2_lighter_mask], 
                                       h2_array[h2_lighter_mask], 
                                       liion_array[h2_lighter_mask],
                                       alpha=0.2, color='blue', 
                                       label='H₂ Advantage')
                if np.any(liion_lighter_mask):
                    ax_be.fill_between(ranges_km[liion_lighter_mask], 
                                       liion_array[liion_lighter_mask], 
                                       h2_array[liion_lighter_mask],
                                       alpha=0.2, color='red', 
                                       label='Li-ion Advantage')
                
                ax_be.set_xlabel('Mission Range (km)', fontsize=11, fontweight='bold')
                ax_be.set_ylabel('System Mass (kg)', fontsize=11, fontweight='bold')
                ax_be.set_title('Break-Even Analysis: H₂ vs. Li-ion', 
                               fontsize=12, fontweight='bold', pad=10)
                ax_be.grid(True, alpha=0.3, linestyle='--')
                ax_be.legend(fontsize=9, loc='best', framealpha=0.9)
                ax_be.set_xlim([ranges_km[0], ranges_km[-1]])
                y_max = max(max(h2_masses), max(liion_masses)) * 1.1
                ax_be.set_ylim([0, y_max])
                
                plt.tight_layout()
                st.pyplot(fig_break_even, use_container_width=True)
        
        with row1_col2:
            with st.expander("📊 Break-Even Metrics", expanded=True):
                if break_even_range is not None:
                    st.metric("Break-Even Range", f"{break_even_range:.1f} km")
                    st.metric("System Mass", f"{break_even_mass:.2f} kg")
                    
                    short_range = 50
                    long_range = 250
                    h2_short = calculate_h2_system_mass_for_range(short_range, current_tank_weight)
                    liion_short = calculate_liion_system_mass_for_range(short_range)
                    mass_diff_short = h2_short - liion_short
                    h2_long = calculate_h2_system_mass_for_range(long_range, current_tank_weight)
                    liion_long = calculate_liion_system_mass_for_range(long_range)
                    mass_diff_long = h2_long - liion_long
                    
                    st.markdown("**Mass Difference:**")
                    st.metric(f"At {short_range} km", f"{mass_diff_short:+.2f} kg")
                    st.metric(f"At {long_range} km", f"{mass_diff_long:+.2f} kg")
                    
                    if break_even_range < 100:
                        st.success(f"✅ Viable at {break_even_range:.1f} km")
                    elif break_even_range < 200:
                        st.info(f"ℹ️ Optimal for medium-long range")
                    else:
                        st.warning(f"⚠️ Consider optimization")
                else:
                    st.warning("Break-even not found")
        
        # Bottom section: Additional insights
        st.markdown("---")
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            with st.expander("💡 Key Insights", expanded=False):
                if break_even_range is not None:
                    advantage_ranges = [100, 200, 300]
                    st.markdown("**Mass Advantage:**")
                    for r in advantage_ranges:
                        h2_m = calculate_h2_system_mass_for_range(r, current_tank_weight)
                        liion_m = calculate_liion_system_mass_for_range(r)
                        advantage = ((liion_m - h2_m) / liion_m) * 100
                        if advantage > 0:
                            st.markdown(f"- At {r} km: H₂ is **{advantage:.1f}% lighter**")
                        else:
                            st.markdown(f"- At {r} km: Li-ion is **{abs(advantage):.1f}% lighter**")
        
        with summary_col2:
            with st.expander("📐 Methodology", expanded=False):
                st.markdown("""
                **Break-Even Calculation:**
                1. Energy requirement per range
                2. System mass = Payload + Empty + Tank/Fuel or Battery
                3. Find range where masses equal
                4. Above = H₂ advantage, Below = Li-ion advantage
                """)
    
    with tab3:
        st.header("📊 Reliability & Risk Analysis Dashboard")
        st.markdown("---")
        
        # Parameters section - compact
        with st.expander("⚙️ Simulation Parameters", expanded=False):
            col_mc1, col_mc2 = st.columns(2)
        
        with col_mc1:
            st.subheader("Simulation Parameters")
            base_efficiency = st.slider(
                "Base Efficiency (mean)",
                min_value=0.3,
                max_value=0.7,
                value=0.5,
                step=0.01,
                help="Mean system efficiency for Monte Carlo analysis. Physics: η_system = η_fuel_cell × η_motor × η_propulsive. Typical range: 0.4-0.6. Higher efficiency → More range for same fuel mass."
            )
            
            efficiency_sigma = st.slider(
                "Efficiency Std Dev (σ)",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                help="Standard deviation for efficiency distribution (5% = 0.05). Physics: Represents manufacturing tolerances and operating condition variations. Higher σ → More uncertainty → Wider range distribution."
            )
            
            n_simulations = st.selectbox(
                "Number of Simulations",
                options=[500, 1000, 2000, 5000],
                index=0,  # Default to 500 for stochastic analysis
                help="More simulations = more accurate but slower"
            )
        
        with col_mc2:
            st.subheader("Boil-off Parameters")
            # Calculate base boil-off rate from tank
            tank = LH2Tank(tank_radius=0.3, tank_length=1.5)
            tank.mli_thickness = insulation_thickness_mm / 1000.0
            base_bor = tank.calculate_boil_off_rate() / 3600  # Convert to kg/h
            
            st.metric("Base Boil-off Rate", f"{base_bor*3600:.4f} kg/h",
                     help="Calculated from current tank configuration")
            
            boil_off_sigma = st.slider(
                "Boil-off Std Dev (σ)",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                help="Standard deviation for boil-off rate distribution. Physics: BOG rate varies with ambient temperature, MLI aging, and tank orientation. Higher σ → More fuel loss uncertainty → Lower P90 range."
            )
            
            mission_duration = st.slider(
                "Mission Duration (hours)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Mission duration for boil-off calculation. Physics: Total BOG loss = BOG_rate × time. Longer missions → More cumulative fuel loss → Reduced available fuel for propulsion."
            )
        
        # Run Stochastic Analysis (default) or Monte Carlo Analysis
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔄 Run Stochastic Analysis (500 iterations)", type="primary"):
                with st.spinner(f"Running 500 stochastic iterations..."):
                    stoch_results = stochastic_mission_analysis(
                        base_efficiency=base_efficiency,
                        base_boil_off_rate=base_bor,
                        fuel_mass=2.0,
                        payload_mass=5.0,
                        empty_mass=10.0,
                        lift_to_drag=15.0,
                        mission_duration_hours=mission_duration,
                        n_iterations=500,
                        seed=42  # For reproducibility
                    )
                    
                    st.session_state['stoch_results'] = stoch_results
                    st.session_state['analysis_type'] = 'stochastic'
        
        with col_btn2:
            if st.button("🔄 Run Extended Monte Carlo Analysis", type="secondary"):
                with st.spinner(f"Running {n_simulations} Monte Carlo simulations..."):
                    mc_results = monte_carlo_mission_analysis(
                        base_efficiency=base_efficiency,
                        base_boil_off_rate=base_bor,
                        fuel_mass=2.0,
                        payload_mass=5.0,
                        empty_mass=10.0,
                        lift_to_drag=15.0,
                        mission_duration_hours=mission_duration,
                        n_simulations=n_simulations,
                        efficiency_sigma=efficiency_sigma,
                        boil_off_sigma=boil_off_sigma,
                        seed=42  # For reproducibility
                    )
                    
                    st.session_state['mc_results'] = mc_results
                    st.session_state['analysis_type'] = 'monte_carlo'
        
        # Display results if available
        if 'stoch_results' in st.session_state:
            stoch_results = st.session_state['stoch_results']
            
            # Dashboard layout
            st.markdown("---")
            row1_col1, row1_col2 = st.columns(2)
            
            with row1_col1:
                with st.expander("📈 Range Distribution", expanded=True):
                    fig_stoch1, ax1 = plt.subplots(figsize=(6, 4))
            
            # Histogram of mission ranges
            ax1.hist(stoch_results['ranges_km'], bins=50, color='steelblue', 
                    alpha=0.7, edgecolor='black', linewidth=1.2)
            
            # Add statistical lines
            mean_range = stoch_results['mean_range']
            std_range = stoch_results['std_range']
            percentiles = stoch_results['percentiles']
            
            ax1.axvline(mean_range, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_range:.2f} km')
            ax1.axvline(percentiles['p50'], color='green', linestyle='--', linewidth=2,
                       label=f'Median (P50): {percentiles["p50"]:.2f} km')
            ax1.axvline(percentiles['p90'], color='gold', linestyle='-', linewidth=3,
                       label=f'P90 Range: {percentiles["p90"]:.2f} km', zorder=5)
            ax1.axvline(percentiles['p10'], color='orange', linestyle=':', linewidth=2,
                       label=f'P10: {percentiles["p10"]:.2f} km')
            
            # Fill confidence intervals
            ax1.axvspan(percentiles['p10'], percentiles['p90'], 
                       alpha=0.15, color='gold', label='80% Confidence Interval (P10-P90)')
            ax1.axvspan(percentiles['p25'], percentiles['p75'], 
                       alpha=0.2, color='green', label='50% Confidence Interval (P25-P75)')
            
            # Highlight P90 region
            ax1.axvspan(percentiles['p90'], stoch_results['max_range'], 
                       alpha=0.1, color='red', label='Top 10% (Exceeds P90)')
            
            ax1.set_xlabel('Mission Range (km)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax1.set_title('Stochastic Analysis: Mission Range Distribution\n' +
                         f'Probability Distribution from {stoch_results["n_iterations"]} Iterations\n' +
                         f'Fuel Cell Efficiency: {stoch_results["efficiency_variation"]}, ' +
                         f'Boil-off Rate: {stoch_results["boil_off_variation"]}',
                         fontsize=14, fontweight='bold', pad=15)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(fontsize=9, loc='upper right')
            
            # Distribution of efficiencies used
            ax2.hist(stoch_results['efficiencies'], bins=30, color='coral', 
                    alpha=0.7, edgecolor='black', linewidth=1.2)
            ax2.axvline(base_efficiency, color='red', linestyle='--', linewidth=2,
                       label=f'Base: {base_efficiency:.3f}')
            ax2.axvline(base_efficiency * 1.03, color='orange', linestyle=':', linewidth=1.5,
                       label=f'+3%: {base_efficiency * 1.03:.3f}')
            ax2.axvline(base_efficiency * 0.97, color='orange', linestyle=':', linewidth=1.5,
                       label=f'-3%: {base_efficiency * 0.97:.3f}')
            ax2.set_xlabel('System Efficiency', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax2.set_title('Fuel Cell Efficiency Distribution\n(Gaussian, ±3% variation)',
                         fontsize=14, fontweight='bold', pad=15)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig_stoch)
            
            # Update results reference for display below
            results = stoch_results
            analysis_type = 'stochastic'
            
        elif 'mc_results' in st.session_state:
            mc_results = st.session_state['mc_results']
            
            # Create histogram plot
            fig_mc, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Histogram of mission ranges
            ax1.hist(mc_results['ranges_km'], bins=50, color='steelblue', 
                    alpha=0.7, edgecolor='black', linewidth=1.2)
            
            # Add statistical lines
            mean_range = mc_results['mean_range']
            std_range = mc_results['std_range']
            percentiles = mc_results['percentiles']
            
            ax1.axvline(mean_range, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_range:.2f} km')
            ax1.axvline(percentiles['p50'], color='green', linestyle='--', linewidth=2,
                       label=f'Median: {percentiles["p50"]:.2f} km')
            ax1.axvline(percentiles['p5'], color='orange', linestyle=':', linewidth=2,
                       label=f'5th percentile: {percentiles["p5"]:.2f} km')
            ax1.axvline(percentiles['p95'], color='orange', linestyle=':', linewidth=2,
                       label=f'95th percentile: {percentiles["p95"]:.2f} km')
            
            # Fill confidence intervals
            ax1.axvspan(percentiles['p25'], percentiles['p75'], 
                       alpha=0.2, color='green', label='50% Confidence Interval')
            ax1.axvspan(percentiles['p5'], percentiles['p95'], 
                       alpha=0.1, color='orange', label='90% Confidence Interval')
            
            ax1.set_xlabel('Mission Range (km)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax1.set_title('Monte Carlo Analysis: Mission Range Distribution\n' +
                         f'Probability Distribution from {mc_results["n_simulations"]} Simulations',
                         fontsize=14, fontweight='bold', pad=15)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(fontsize=10, loc='upper right')
            
            # Distribution of efficiencies used
            ax2.hist(mc_results['efficiencies'], bins=30, color='coral', 
                    alpha=0.7, edgecolor='black', linewidth=1.2)
            ax2.axvline(base_efficiency, color='red', linestyle='--', linewidth=2,
                       label=f'Base: {base_efficiency:.3f}')
            ax2.set_xlabel('System Efficiency', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax2.set_title('Efficiency Distribution\n(Gaussian, σ = {:.3f})'.format(efficiency_sigma),
                         fontsize=14, fontweight='bold', pad=15)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig_mc)
            
            # Update results reference for display below
            results = mc_results
            analysis_type = 'monte_carlo'
        
        # Display statistics if results are available
        if 'stoch_results' in st.session_state or 'mc_results' in st.session_state:
            if 'stoch_results' in st.session_state:
                results = st.session_state['stoch_results']
                analysis_type = 'stochastic'
            else:
                results = st.session_state['mc_results']
                analysis_type = 'monte_carlo'
            
            # Display statistics
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.subheader("📈 Range Statistics")
                st.metric("Mean Range", f"{results['mean_range']:.2f} km")
                st.metric("Std Deviation", f"{results['std_range']:.2f} km")
                st.metric("Coefficient of Variation", f"{(results['std_range']/results['mean_range'])*100:.2f}%")
                st.metric("Min Range", f"{results['min_range']:.2f} km")
                st.metric("Max Range", f"{results['max_range']:.2f} km")
            
            with col_stat2:
                st.subheader("📊 Percentiles & Reliability Metrics")
                if analysis_type == 'stochastic':
                    st.metric("**P90 Range** (90% Probability)", 
                             f"{results['p90_range']:.2f} km",
                             help="Industry standard: Range the UAV is 90% likely to achieve",
                             delta=None)
                    st.metric("P10 Range", f"{results['p10_range']:.2f} km",
                             help="10% of missions achieve less than this range")
                else:
                    st.metric("5th Percentile", f"{results['percentiles']['p5']:.2f} km",
                             help="5% of missions achieve less than this range")
                
                st.metric("25th Percentile", f"{results['percentiles']['p25']:.2f} km",
                         help="25% of missions achieve less than this range")
                st.metric("50th Percentile (Median)", f"{results['percentiles']['p50']:.2f} km",
                         help="50% of missions achieve less than this range")
                st.metric("75th Percentile", f"{results['percentiles']['p75']:.2f} km",
                         help="75% of missions achieve less than this range")
                if 'p90' in results['percentiles']:
                    st.metric("90th Percentile (P90)", f"{results['percentiles']['p90']:.2f} km",
                             help="90% of missions achieve less than this range")
                st.metric("95th Percentile", f"{results['percentiles']['p95']:.2f} km",
                         help="95% of missions achieve less than this range")
            
            with col_stat3:
                st.subheader("⚠️ Risk Assessment")
                # Calculate probability of mission success for different range requirements
                target_ranges = [100, 150, 200, 250]
                st.markdown("**Mission Success Probability:**")
                n_sims = results.get('n_iterations', results.get('n_simulations', 500))
                for target in target_ranges:
                    success_prob = (results['ranges_km'] >= target).sum() / n_sims * 100
                    if success_prob >= 90:
                        st.success(f"≥{target} km: **{success_prob:.1f}%** ✅")
                    elif success_prob >= 70:
                        st.info(f"≥{target} km: **{success_prob:.1f}%** ⚠️")
                    else:
                        st.warning(f"≥{target} km: **{success_prob:.1f}%** ❌")
                
                # Reliability metrics
                st.markdown("---")
                st.markdown("**Reliability Metrics:**")
                cv = (results['std_range'] / results['mean_range']) * 100
                if cv < 5:
                    st.success(f"Low Variability (CV = {cv:.2f}%)")
                elif cv < 10:
                    st.info(f"Moderate Variability (CV = {cv:.2f}%)")
                else:
                    st.warning(f"High Variability (CV = {cv:.2f}%)")
                
                # P90 Range significance
                if analysis_type == 'stochastic':
                    st.markdown("---")
                    st.markdown("**P90 Range Significance:**")
                    p90_margin = ((results['mean_range'] - results['p90_range']) / results['mean_range']) * 100
                    if p90_margin < 5:
                        st.success(f"P90 is {abs(p90_margin):.1f}% below mean - Excellent reliability")
                    elif p90_margin < 10:
                        st.info(f"P90 is {abs(p90_margin):.1f}% below mean - Good reliability")
                    else:
                        st.warning(f"P90 is {abs(p90_margin):.1f}% below mean - Consider design improvements")
            
            # Detailed explanation
            st.markdown("---")
            st.subheader("📐 Analysis Methodology")
            if analysis_type == 'stochastic':
                st.markdown("""
                **Stochastic Analysis Process:**
                
                1. **Parameter Sampling**: For each iteration (500 total), sample:
                   - **Fuel Cell Efficiency**: Gaussian distribution with ±3% variation (σ = 0.03)
                   - **Tank Boil-off Rate**: Gaussian distribution with ±5% variation (σ = 0.05)
                
                2. **Fuel Loss Calculation**: Account for boil-off losses during mission:
                   Fuel Loss = Boil-off Rate × Mission Duration
                
                3. **Mission Simulation**: Run complete mission profile with sampled parameters.
                
                4. **Statistical Analysis**: Calculate mean, standard deviation, and percentiles
                   from all iteration results.
                
                5. **P90 Range Calculation**: Determine the 90th percentile range - the range
                   the UAV is 90% likely to achieve (industry standard safety metric).
                
                6. **Risk Assessment**: Determine probability of achieving target mission ranges.
                
                **Key Insights:**
                - The histogram shows the probability distribution of achievable mission ranges
                - **P90 Range** is the industry standard safety metric for mission planning
                - Confidence intervals (P10-P90) represent the expected range of outcomes
                - Higher variability indicates greater uncertainty and risk
                - This analysis enables informed decision-making under uncertainty
                
                **Engineering Significance:**
                The P90 Range provides a conservative estimate for mission planning, ensuring
                that 90% of missions will achieve at least this range. This is critical for:
                - Flight planning and route selection
                - Fuel loading decisions
                - Safety margin calculations
                - Regulatory compliance and certification
                """)
            else:
                st.markdown("""
                **Monte Carlo Stochastic Analysis Process:**
                
                1. **Parameter Sampling**: For each simulation, sample efficiency and boil-off rate
                   from Gaussian distributions centered on base values.
                
                2. **Fuel Loss Calculation**: Account for boil-off losses during mission:
                   Fuel Loss = Boil-off Rate × Mission Duration
                
                3. **Mission Simulation**: Run complete mission profile with sampled parameters.
                
                4. **Statistical Analysis**: Calculate mean, standard deviation, and percentiles
                   from all simulation results.
                
                5. **Risk Assessment**: Determine probability of achieving target mission ranges.
                
                **Key Insights:**
                - The histogram shows the probability distribution of achievable mission ranges
                - Confidence intervals (5th-95th percentile) represent the expected range of outcomes
                - Higher variability indicates greater uncertainty and risk
                - This analysis enables informed decision-making under uncertainty
                """)
        else:
            st.info("👆 Click 'Run Stochastic Analysis (500 iterations)' or 'Run Extended Monte Carlo Analysis' to generate reliability and risk assessment")
    
    with tab4:
        st.header("🛡️ Safety & Compliance Analysis")
        st.markdown("""
        **EASA Standards Compliance for Hydrogen Storage Systems**
        
        This module implements safety compliance checks based on European Union Aviation
        Safety Agency (EASA) standards for hydrogen storage in aerospace applications.
        
        **Key Safety Parameters:**
        - **Factor of Safety (FoS)**: Minimum 2.2 per EASA standards
        - **Burst Pressure**: Calculated using thin-walled pressure vessel theory
        - **Energy Buffer**: Reserve fuel for emergency landing scenarios
        """)
        
        col_safety1, col_safety2 = st.columns(2)
        
        with col_safety1:
            st.subheader("📐 Tank Geometry")
            tank_radius_safety = st.slider(
                "Tank Radius (m)",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="safety_radius",
                help="Tank radius. Physics: Hoop stress σ = PD/2t. Larger radius → Higher stress at same pressure → Requires thicker walls or stronger material for same FoS."
            )
            
            tank_length_safety = st.slider(
                "Tank Length (m)",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.1,
                key="safety_length",
                help="Tank length. Physics: Volume V = πr²L. Longer tanks increase volume but also surface area (heat leak). Affects structural loading and thermal performance."
            )
            
            wall_thickness_safety_mm = st.slider(
                "Wall Thickness (mm)",
                min_value=1.0,
                max_value=10.0,
                value=2.0,
                step=0.1,
                key="safety_thickness",
                help="Wall thickness. Physics: Hoop stress σ = PD/2t. Thicker walls → Lower stress → Higher FoS but increased mass. EASA requires FoS ≥ 2.2."
            )
            wall_thickness_safety = wall_thickness_safety_mm / 1000.0
        
        with col_safety2:
            st.subheader("⚙️ Operating Conditions")
            operating_pressure_safety_kpa = st.slider(
                "Operating Pressure (kPa)",
                min_value=100,
                max_value=1000,
                value=500,
                step=50,
                key="safety_pressure",
                help="Operating pressure. Physics: Hoop stress σ = PD/2t. Higher pressure → Higher stress → Requires thicker walls or stronger material. Typical LH2 storage: 200-500 kPa."
            )
            operating_pressure_safety = operating_pressure_safety_kpa * 1000
            
            material_strength_mpa = st.slider(
                "Material Yield Strength (MPa)",
                min_value=200,
                max_value=500,
                value=350,
                step=10,
                help="Material yield strength. Physics: FoS = σ_yield / σ_operating. Higher strength → Higher FoS for same wall thickness. Typical: 350 MPa (aerospace aluminum), 450 MPa (titanium)."
            )
            material_yield_strength = material_strength_mpa * 1e6
        
        # Initialize safety compliance checker
        safety_checker = SafetyCompliance(
            tank_radius=tank_radius_safety,
            tank_length=tank_length_safety,
            wall_thickness=wall_thickness_safety,
            material_yield_strength=material_yield_strength,
            operating_pressure=operating_pressure_safety
        )
        
        # Check EASA compliance
        compliance_result = safety_checker.check_easa_compliance()
        
        # Display compliance status prominently
        st.markdown("---")
        col_status1, col_status2, col_status3 = st.columns(3)
        
        with col_status1:
            if compliance_result['is_compliant']:
                st.success("✅ **EASA COMPLIANT**")
            else:
                st.error("❌ **NON-COMPLIANT**")
                st.warning("⚠️ Factor of Safety below EASA minimum requirement!")
        
        with col_status2:
            st.metric(
                "Factor of Safety",
                f"{compliance_result['factor_of_safety']:.2f}",
                delta=f"{compliance_result['margin']:.2f}" if compliance_result['margin'] >= 0 else f"{compliance_result['margin']:.2f}",
                delta_color="normal" if compliance_result['is_compliant'] else "inverse",
                help=f"EASA Minimum: {compliance_result['easa_min_fos']:.1f}"
            )
        
        with col_status3:
            st.metric(
                "Safety Margin",
                f"{(compliance_result['factor_of_safety'] / compliance_result['easa_min_fos'] - 1) * 100:.1f}%",
                help="Margin above EASA minimum requirement"
            )
        
        # Detailed pressure vessel analysis
        st.markdown("---")
        st.subheader("🔬 Pressure Vessel Analysis")
        
        col_pv1, col_pv2 = st.columns(2)
        
        with col_pv1:
            st.markdown("**Thin-Walled Pressure Vessel Formula:**")
            st.latex(r"\sigma = \frac{PD}{2t}")
            st.markdown("where:")
            st.markdown("- σ = Hoop stress (Pa)")
            st.markdown("- P = Internal pressure (Pa)")
            st.markdown("- D = Diameter (m)")
            st.markdown("- t = Wall thickness (m)")
            
            st.markdown("**Burst Pressure:**")
            st.latex(r"P_{burst} = \frac{2t \times \sigma_{yield}}{D}")
        
        with col_pv2:
            st.markdown("**Calculated Values:**")
            st.metric("Tank Diameter", f"{2 * tank_radius_safety:.3f} m")
            st.metric("Wall Thickness", f"{wall_thickness_safety_mm:.2f} mm")
            st.metric("Operating Pressure", f"{operating_pressure_safety_kpa:.1f} kPa")
            st.metric("Burst Pressure", f"{compliance_result['burst_pressure_kPa']:.1f} kPa")
            st.metric("Material Yield Strength", f"{material_strength_mpa:.0f} MPa")
            
            # Calculate hoop stress at operating pressure
            diameter = 2 * tank_radius_safety
            hoop_stress = (operating_pressure_safety * diameter) / (2 * wall_thickness_safety)
            hoop_stress_mpa = hoop_stress / 1e6
            st.metric("Hoop Stress (Operating)", f"{hoop_stress_mpa:.1f} MPa")
        
        # Energy Buffer Calculation
        st.markdown("---")
        st.subheader("⛽ Energy Buffer (Reserve Fuel)")
        st.markdown("""
        **EASA Reserve Fuel Requirements:**
        
        Aircraft must maintain sufficient fuel to:
        1. Divert to alternate airport if destination is closed
        2. Hold in pattern (typically 30 minutes)
        3. Execute final approach and landing
        
        This analysis calculates the minimum fuel required for safe emergency landing.
        """)
        
        col_eb1, col_eb2 = st.columns(2)
        
        with col_eb1:
            reserve_time_safety = st.slider(
                "Reserve Flight Time (minutes)",
                min_value=15,
                max_value=60,
                value=30,
                step=5,
                key="safety_reserve_time",
                help="Reserve flight time. Physics: Reserve fuel = Power × time / (η × LHV). EASA standard: 30 minutes reserve. Longer reserve → Higher safety margin → More fuel required → Reduced mission range."
            )
            
            # Get current mission parameters
            try:
                current_range_safety = calculate_range_with_tank_weight(
                    calculate_tank_weight_from_thickness(insulation_thickness_mm),
                    fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                    lift_to_drag=15.0, total_efficiency=0.5, cruise_velocity=cruise_velocity
                )
            except ValueError as e:
                st.error(f"⚠️ Cannot calculate range: {str(e)}")
                st.info("💡 Tip: Ensure fuel mass > 0, empty mass > 0, and cruise velocity > 0")
                current_range_safety = 0.0
        
        with col_eb2:
            st.markdown("**Mission Parameters:**")
            st.metric("Cruise Range", f"{current_range_safety:.2f} km")
            st.metric("Cruise Velocity", f"{cruise_velocity:.1f} m/s")
            st.metric("L/D Ratio", "15.0")
            st.metric("System Efficiency", "50%")
        
        # Calculate energy buffer
        energy_buffer_result = safety_checker.calculate_energy_buffer(
            cruise_range_km=current_range_safety,
            reserve_time_minutes=reserve_time_safety,
            cruise_velocity=cruise_velocity,
            lift_to_drag=15.0,
            total_efficiency=0.5
        )
        
        min_safe_fuel_result = safety_checker.calculate_minimum_safe_fuel(
            total_fuel_mass=2.0,
            cruise_range_km=current_range_safety,
            reserve_time_minutes=reserve_time_safety,
            cruise_velocity=cruise_velocity,
            lift_to_drag=15.0,
            total_efficiency=0.5
        )
        
        # Display energy buffer results
        col_eb3, col_eb4, col_eb5 = st.columns(3)
        
        with col_eb3:
            st.subheader("📊 Reserve Fuel")
            st.metric("Reserve Fuel Required", 
                     f"{energy_buffer_result['reserve_fuel_mass_kg']*1000:.1f} g",
                     help="Minimum fuel for emergency landing")
            st.metric("Reserve Distance", 
                     f"{energy_buffer_result['reserve_distance_km']:.2f} km",
                     help="Distance covered during reserve flight")
            st.metric("Reserve Time", 
                     f"{reserve_time_safety} minutes",
                     help="EASA standard reserve flight time")
        
        with col_eb4:
            st.subheader("✈️ Usable Fuel")
            st.metric("Total Fuel", "2000.0 g", help="Total fuel capacity")
            st.metric("Usable Fuel", 
                     f"{min_safe_fuel_result['usable_fuel_mass_kg']*1000:.1f} g",
                     help="Fuel available for mission (excluding reserve)")
            st.metric("Usable Range", 
                     f"{min_safe_fuel_result['usable_range_km']:.2f} km",
                     help="Mission range with usable fuel")
        
        with col_eb5:
            st.subheader("⚠️ Safety Thresholds")
            st.metric("Warning Threshold", 
                     f"{min_safe_fuel_result['warning_threshold_kg']*1000:.1f} g",
                     help="Fuel level to trigger warning")
            st.metric("Critical Threshold", 
                     f"{min_safe_fuel_result['critical_threshold_kg']*1000:.1f} g",
                     help="Fuel level requiring immediate landing")
            st.metric("Reserve Percentage", 
                     f"{min_safe_fuel_result['reserve_percentage']:.1f}%",
                     help="Reserve fuel as percentage of total")
        
        # Visual warning if non-compliant
        if not compliance_result['is_compliant']:
            st.error("""
            ⚠️ **SAFETY WARNING: NON-COMPLIANT DESIGN**
            
            The current tank design does not meet EASA Factor of Safety requirements.
            **Recommended Actions:**
            1. Increase wall thickness
            2. Use higher strength material
            3. Reduce operating pressure
            4. Increase tank diameter (if possible)
            
            **Current FoS:** {:.2f} | **Required:** ≥ {:.1f}
            """.format(compliance_result['factor_of_safety'], compliance_result['easa_min_fos']))
        
        # Energy buffer recommendation
        st.info(f"""
        💡 **Energy Buffer Recommendation:**
        
        {energy_buffer_result['recommendation']}
        
        **Safety Guidelines:**
        - Never allow fuel to drop below **{min_safe_fuel_result['warning_threshold_kg']*1000:.1f} g**
        - At **{min_safe_fuel_result['critical_threshold_kg']*1000:.1f} g**, initiate immediate landing procedures
        - Reserve fuel ensures safe diversion to alternate airport if destination is closed
        """)
        
        # Detailed explanation
        st.markdown("---")
        st.subheader("📐 Safety Analysis Methodology")
        st.markdown("""
        **EASA Compliance Check:**
        
        1. **Burst Pressure Calculation**: Uses thin-walled pressure vessel formula
           to determine maximum pressure before failure.
        
        2. **Factor of Safety**: Ratio of burst pressure to operating pressure.
           EASA requires minimum FoS of 2.2 for hydrogen storage systems.
        
        3. **Energy Buffer Calculation**: Determines minimum fuel required for:
           - Alternate airport diversion
           - Holding pattern (30 minutes standard)
           - Final approach and landing
        
        **Safety Margins:**
        - FoS accounts for material variations, manufacturing tolerances, and aging
        - Energy buffer ensures safe operation under emergency conditions
        - Multiple safety thresholds provide progressive warnings
        
        **Critical Importance:**
        These calculations are essential for certification and operational safety
        in hydrogen-electric aircraft systems.
        """)
    
    # Smart Assistant Section (available in all tabs)
    st.markdown("---")
    st.header("🤖 Propulsion System Assistant")
    st.markdown("""
    **AI-Powered Analysis and Recommendations**
    
    Ask questions about your system configuration and receive intelligent recommendations
    based on current simulation results. The assistant analyzes all parameters including
    boil-off rates, safety compliance, mission range, and efficiency.
    """)
    
    # Initialize assistant (can be configured with API key for real LLM)
    assistant = PropulsionAssistant(api_key=None)  # Set API key for real LLM integration
    
    # Text input for user query
    user_query = st.text_input(
        "Ask the Propulsion Assistant",
        placeholder="e.g., 'Is my boil-off rate too high for a 5-hour mission?' or 'How can I improve my Factor of Safety?'",
        key="assistant_query",
        help="Enter your propulsion system question. The AI assistant will analyze your current configuration and provide recommendations."
    )
    
    # Validate text input
    if user_query:
        if len(user_query.strip()) == 0:
            st.warning("⚠️ Please enter a non-empty query.")
            user_query = None
        elif len(user_query) > 1000:
            st.warning("⚠️ Query is too long. Please keep it under 1000 characters.")
            user_query = user_query[:1000]  # Truncate
    
    if user_query:
        # Collect current simulation state
        try:
            tank_weight_current = calculate_tank_weight_from_thickness(insulation_thickness_mm)
            current_range_assistant = calculate_range_with_tank_weight(
                tank_weight_current, fuel_mass=2.0, payload_mass=5.0, empty_mass=10.0,
                lift_to_drag=15.0, total_efficiency=0.5, cruise_velocity=cruise_velocity
            )
        except ValueError as e:
            st.error(f"⚠️ Cannot calculate range: {str(e)}")
            st.info("💡 Tip: Ensure fuel mass > 0, empty mass > 0, and cruise velocity > 0")
            current_range_assistant = 0.0
            tank_weight_current = 0.5
        
        # Get tank and safety parameters
        tank_assistant = LH2Tank(tank_radius=0.3, tank_length=1.5)
        tank_assistant.mli_thickness = insulation_thickness_mm / 1000.0
        heat_leak_assistant = tank_assistant.calculate_total_heat_leak()
        boil_off_rate_assistant = tank_assistant.calculate_boil_off_rate()
        
        # Initialize safety checker for FoS
        safety_assistant = SafetyCompliance(
            tank_radius=0.3,
            tank_length=1.5,
            wall_thickness=0.002,  # Default 2mm
            material_yield_strength=350e6,
            operating_pressure=500e3
        )
        compliance_assistant = safety_assistant.check_easa_compliance()
        
        # Calculate energy buffer
        energy_buffer_assistant = safety_assistant.calculate_energy_buffer(
            cruise_range_km=current_range_assistant,
            reserve_time_minutes=30,
            cruise_velocity=cruise_velocity,
            lift_to_drag=15.0,
            total_efficiency=0.5
        )
        
        min_safe_fuel_assistant = safety_assistant.calculate_minimum_safe_fuel(
            total_fuel_mass=2.0,
            cruise_range_km=current_range_assistant,
            reserve_time_minutes=30,
            cruise_velocity=cruise_velocity,
            lift_to_drag=15.0,
            total_efficiency=0.5
        )
        
        # Prepare simulation state
        simulation_state = {
            'tank': {
                'radius': 0.3,
                'length': 1.5,
                'wall_thickness_mm': 2.0,
                'insulation_thickness_mm': insulation_thickness_mm,
                'operating_pressure_kpa': 500.0,
                'material_strength_mpa': 350.0,
                'weight': tank_weight_current
            },
            'fuel_cell': {
                'power_kw': fuel_cell_power_kw,
                'e_ocv': 1.2,
                'temperature': 353.15
            },
            'mission': {
                'payload_mass': 5.0,
                'fuel_mass': 2.0,
                'cruise_velocity': cruise_velocity,
                'lift_to_drag': 15.0,
                'total_efficiency': 0.5
            },
            'safety': {
                'fos': compliance_assistant['factor_of_safety'],
                'is_compliant': compliance_assistant['is_compliant']
            },
            'results': {
                'range_km': current_range_assistant,
                'heat_leak': heat_leak_assistant,
                'boil_off_rate': boil_off_rate_assistant / 3600,  # Convert to kg/s
                'boil_off_rate_g_per_h': boil_off_rate_assistant,
                'fos': compliance_assistant['factor_of_safety'],
                'reserve_fuel_g': energy_buffer_assistant['reserve_fuel_mass_kg'] * 1000,
                'usable_range_km': min_safe_fuel_assistant['usable_range_km']
            }
        }
        
        # Get assistant response
        with st.spinner("🤔 Analyzing system configuration and generating recommendations..."):
            assistant_response = assistant.analyze_and_recommend(user_query, simulation_state)
        
        # Display response in styled box
        st.success("""
        **🤖 AI Propulsion Consultant Response**
        
        """ + assistant_response)
        
        # Store assistant response and simulation state for PDF generation
        st.session_state['assistant_response'] = assistant_response
        st.session_state['simulation_state_for_pdf'] = simulation_state
        
        # Export Technical Memo button
        st.markdown("---")
        col_export1, col_export2 = st.columns([2, 1])
        
        with col_export1:
            st.markdown("**📄 Export Technical Memorandum**")
            st.markdown("Generate a formal NASA-style technical memo PDF with Executive Summary, "
                       "optimized parameters table, and Risks & Mitigations analysis.")
        
        with col_export2:
            if not PDF_GENERATOR_AVAILABLE:
                st.warning("⚠️ PDF Export Unavailable")
                st.info("""
                **ReportLab is not installed.**
                
                To enable PDF export:
                ```bash
                pip install reportlab>=3.6.0
                ```
                
                Then restart the application.
                """)
            elif st.button("📥 Export Technical Memo", type="primary", use_container_width=True):
                try:
                    with st.spinner("Generating PDF technical memorandum..."):
                        # Generate PDF
                        pdf_bytes = generate_nasa_technical_memo(
                            simulation_state=simulation_state,
                            assistant_response=assistant_response
                        )
                        
                        if pdf_bytes is None:
                            st.error("❌ PDF generation returned None")
                            st.stop()
                        
                        # Create download button
                        filename = f"NASA_TechMemo_H2_UAV_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True
                        )
                        
                        st.success("✅ PDF generated successfully! Click the download button above.")
                        
                except ImportError as e:
                    st.error(f"❌ ReportLab Import Error: {str(e)}")
                    st.info("💡 Please install ReportLab: `pip install reportlab>=3.6.0`")
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
                    st.info("💡 Make sure ReportLab is installed: `pip install reportlab>=3.6.0`")
        
        # Option to configure API key
        with st.expander("⚙️ Configure LLM Integration (Optional)"):
            st.markdown("""
            **To use a real LLM API (OpenAI, Anthropic, etc.):**
            
            1. Set your API key in the code: `PropulsionAssistant(api_key="your-key")`
            2. Uncomment the API call logic in `core/assistant.py`
            3. Install required packages: `pip install openai` or `anthropic`
            
            **Current Mode**: Rule-based intelligent analysis (works offline)
            
            The rule-based system provides accurate recommendations based on engineering
            principles and EASA standards. For more advanced analysis, integrate with
            an LLM API.
            """)
    
    with tab5:
        st.header("🏆 Industry Benchmarks Comparison")
        st.markdown("""
        **Compare Your UAV Design Against Industry Leaders**
        
        This module compares your hydrogen-electric UAV's performance metrics against
        established industry benchmarks, including the AeroDelft Phoenix (TU Delft) and
        Airbus ZEROe regional concept aircraft.
        
        **Key Metrics Compared:**
        - **System Efficiency**: Total propulsion efficiency (fuel cell + motor + propulsive)
        - **Energy Density**: Effective energy density including tank/system mass (Wh/kg)
        """)
        
        # Calculate current UAV metrics
        # Get current system parameters from session state or defaults
        current_insulation = st.session_state.get('current_insulation', 50.0)
        fuel_cell_power = st.session_state.get('fuel_cell_power_kw', 5.0)
        cruise_velocity = st.session_state.get('cruise_velocity', 30.0)
        lift_to_drag = st.session_state.get('lift_to_drag', 15.0)
        
        # Calculate tank weight
        tank_weight = calculate_tank_weight_from_thickness(current_insulation)
        
        # Mission parameters
        payload_mass = 5.0  # kg
        fuel_mass = 2.0  # kg
        empty_mass = 10.0  # kg
        total_mass = payload_mass + fuel_mass + empty_mass + tank_weight
        
        # Calculate system efficiency from energy balance
        # Use same values as Energy Balance Analysis section for consistency
        fuel_cell_efficiency = 0.55  # 55% fuel cell efficiency (matches Energy Balance section)
        motor_inverter_efficiency = 0.88  # 88% motor/inverter efficiency (matches Energy Balance section)
        total_system_efficiency = 0.5  # 50% total system efficiency (matches Energy Balance section)
        uav_system_efficiency = total_system_efficiency  # Use total system efficiency
        
        # Calculate effective energy density (Wh/kg) including tank mass
        hydrogen_energy_density_wh_per_kg = 120000 / 3600  # 120 MJ/kg = 33,333 Wh/kg (pure H2)
        # Effective energy density = (fuel_mass * H2_energy_density) / total_system_mass
        uav_energy_density_wh_per_kg = (fuel_mass * hydrogen_energy_density_wh_per_kg) / total_mass
        
        # Display current UAV metrics
        st.markdown("---")
        col_uav1, col_uav2, col_uav3 = st.columns(3)
        
        with col_uav1:
            st.subheader("✈️ Your UAV")
            st.metric("System Efficiency", f"{uav_system_efficiency*100:.1f}%")
            st.metric("Energy Density", f"{uav_energy_density_wh_per_kg:.0f} Wh/kg")
            st.metric("Total Mass", f"{total_mass:.1f} kg")
        
        with col_uav2:
            st.subheader("📊 System Parameters")
            st.metric("Fuel Mass", f"{fuel_mass:.1f} kg")
            st.metric("Tank Weight", f"{tank_weight:.2f} kg")
            st.metric("Payload Mass", f"{payload_mass:.1f} kg")
        
        with col_uav3:
            st.subheader("⚙️ Performance")
            st.metric("Lift-to-Drag", f"{lift_to_drag:.1f}")
            st.metric("Fuel Cell Power", f"{fuel_cell_power:.1f} kW")
            st.metric("Cruise Velocity", f"{cruise_velocity:.1f} m/s")
        
        # Benchmark comparison button
        st.markdown("---")
        if st.button("🔍 Compare with Industry Benchmarks", type="primary", use_container_width=True):
            st.session_state['show_benchmarks'] = True
        
        # Display comparison if button clicked
        if st.session_state.get('show_benchmarks', False):
            benchmarks = get_all_benchmarks()
            
            # Prepare data for comparison chart
            names = ["Your UAV"] + [b.name for b in benchmarks]
            efficiencies = [uav_system_efficiency] + [b.system_efficiency for b in benchmarks]
            energy_densities = [uav_energy_density_wh_per_kg] + [b.energy_density_wh_per_kg for b in benchmarks]
            
            # Create comparison chart
            st.markdown("---")
            st.subheader("📊 Efficiency & Energy Density Comparison")
            
            # Create scatter plot with efficiency vs energy density
            fig_benchmark = go.Figure()
            
            # Improved color scheme with better contrast and professional appearance
            # Your UAV: Vibrant blue with gold accent (stands out)
            # AeroDelft Phoenix: Deep orange/red (warm, professional)
            # Airbus ZEROe: Forest green (corporate, trustworthy)
            colors = {
                'uav': '#1E88E5',  # Bright blue - Your design stands out
                'aerodelft': '#E65100',  # Deep orange - Warm, energetic
                'airbus': '#2E7D32'  # Forest green - Professional, established
            }
            
            # Gradient backgrounds for better visual appeal
            marker_colors = {
                'uav': '#1E88E5',  # Primary blue
                'aerodelft': '#FF6F00',  # Vibrant orange
                'airbus': '#388E3C'  # Rich green
            }
            
            # Your UAV - Make it stand out with star symbol and bright color
            fig_benchmark.add_trace(go.Scatter(
                x=[uav_energy_density_wh_per_kg],
                y=[uav_system_efficiency * 100],
                mode='markers+text',
                name='Your UAV',
                marker=dict(
                    size=30,
                    color=marker_colors['uav'],
                    symbol='star',
                    line=dict(width=3, color='#0D47A1'),  # Darker blue border
                    opacity=0.9
                ),
                text=['★ Your UAV'],
                textposition='top center',
                textfont=dict(size=13, color=marker_colors['uav'], family='Arial, sans-serif'),
                hovertemplate='<b>Your UAV</b><br>Energy Density: %{x:.0f} Wh/kg<br>Efficiency: %{y:.1f}%<extra></extra>'
            ))
            
            # Industry benchmarks with distinct colors and styles
            benchmark_styles = [
                {
                    'name': 'AeroDelft Phoenix',
                    'color': marker_colors['aerodelft'],
                    'symbol': 'square',
                    'size': 22,
                    'border': '#BF360C'  # Darker orange border
                },
                {
                    'name': 'Airbus ZEROe Regional',
                    'color': marker_colors['airbus'],
                    'symbol': 'diamond',
                    'size': 22,
                    'border': '#1B5E20'  # Darker green border
                }
            ]
            
            for i, benchmark in enumerate(benchmarks):
                style = benchmark_styles[i] if i < len(benchmark_styles) else benchmark_styles[0]
                fig_benchmark.add_trace(go.Scatter(
                    x=[benchmark.energy_density_wh_per_kg],
                    y=[benchmark.system_efficiency * 100],
                    mode='markers+text',
                    name=benchmark.name,
                    marker=dict(
                        size=style['size'],
                        color=style['color'],
                        symbol=style['symbol'],
                        line=dict(width=2.5, color=style['border']),
                        opacity=0.85
                    ),
                    text=[benchmark.name],
                    textposition='top center',
                    textfont=dict(size=11, color=style['color'], family='Arial, sans-serif'),
                    hovertemplate=f'<b>{benchmark.name}</b><br>Energy Density: %{{x:.0f}} Wh/kg<br>Efficiency: %{{y:.1f}}%<extra></extra>'
                ))
            
            fig_benchmark.update_layout(
                title={
                    'text': 'System Efficiency vs. Energy Density Comparison<br><sub>Your UAV vs. Industry Benchmarks</sub>',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'family': 'Arial, sans-serif', 'color': '#1a1a1a'}
                },
                xaxis=dict(
                    title=dict(
                        text='Energy Density (Wh/kg)',
                        font=dict(size=14, family='Arial, sans-serif', color='#333')
                    ),
                    tickfont=dict(size=12, family='Arial, sans-serif'),
                    gridcolor='#e0e0e0',
                    gridwidth=1,
                    showgrid=True,
                    zeroline=False,
                    linecolor='#333',
                    linewidth=1
                ),
                yaxis=dict(
                    title=dict(
                        text='System Efficiency (%)',
                        font=dict(size=14, family='Arial, sans-serif', color='#333')
                    ),
                    tickfont=dict(size=12, family='Arial, sans-serif'),
                    gridcolor='#e0e0e0',
                    gridwidth=1,
                    showgrid=True,
                    zeroline=False,
                    linecolor='#333',
                    linewidth=1
                ),
                height=650,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12, family='Arial, sans-serif'),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='#ccc',
                    borderwidth=1
                ),
                plot_bgcolor='#fafafa',
                paper_bgcolor='white',
                font=dict(size=12, family='Arial, sans-serif'),
                hovermode='closest',
                margin=dict(l=80, r=50, t=100, b=60)
            )
            
            st.plotly_chart(fig_benchmark, use_container_width=True)
            
            # Delta analysis for each benchmark
            st.markdown("---")
            st.subheader("📐 Delta Analysis")
            st.markdown("""
            **Delta Analysis**: Detailed comparison explaining the differences between your UAV
            and each industry benchmark, including scale effects and design assumptions.
            """)
            
            # Create tabs for each benchmark
            benchmark_tabs = st.tabs([b.name for b in benchmarks])
            
            for idx, (benchmark, tab) in enumerate(zip(benchmarks, benchmark_tabs)):
                with tab:
                    # Calculate delta analysis
                    delta_results = calculate_delta_analysis(
                        uav_efficiency=uav_system_efficiency,
                        uav_energy_density_wh_per_kg=uav_energy_density_wh_per_kg,
                        uav_mass_kg=total_mass,
                        benchmark=benchmark
                    )
                    
                    # Display metrics comparison
                    col_delta1, col_delta2 = st.columns(2)
                    
                    with col_delta1:
                        st.markdown(f"### {benchmark.name} Specifications")
                        st.markdown(f"**Description**: {benchmark.description}")
                        st.markdown(f"**Source**: {benchmark.source}")
                        
                        st.markdown("**Key Specifications:**")
                        spec_df = pd.DataFrame({
                            'Parameter': [
                                'System Efficiency',
                                'Energy Density',
                                'Total Mass',
                                'Fuel Mass',
                                'Payload Mass',
                                'Max Range',
                                'Max Power',
                                'Lift-to-Drag'
                            ],
                            'Value': [
                                f"{benchmark.system_efficiency*100:.1f}%",
                                f"{benchmark.energy_density_wh_per_kg:.0f} Wh/kg",
                                f"{benchmark.total_mass_kg:.0f} kg",
                                f"{benchmark.fuel_mass_kg:.1f} kg",
                                f"{benchmark.payload_mass_kg:.0f} kg",
                                f"{benchmark.max_range_km:.0f} km",
                                f"{benchmark.max_power_kw:.0f} kW",
                                f"{benchmark.lift_to_drag:.1f}"
                            ]
                        })
                        st.dataframe(spec_df, use_container_width=True, hide_index=True)
                    
                    with col_delta2:
                        st.markdown("### Delta Metrics")
                        
                        # Efficiency delta
                        eff_delta = delta_results['delta_efficiency_pct']
                        st.metric(
                            "Efficiency Delta",
                            f"{eff_delta:+.1f}%",
                            help=f"Your UAV: {uav_system_efficiency*100:.1f}% vs {benchmark.name}: {benchmark.system_efficiency*100:.1f}%"
                        )
                        
                        # Energy density delta
                        ed_delta = delta_results['delta_energy_density_pct']
                        st.metric(
                            "Energy Density Delta",
                            f"{ed_delta:+.1f}%",
                            help=f"Your UAV: {uav_energy_density_wh_per_kg:.0f} Wh/kg vs {benchmark.name}: {benchmark.energy_density_wh_per_kg:.0f} Wh/kg"
                        )
                        
                        # Scale factor
                        scale_factor = delta_results['scale_factor']
                        st.metric(
                            "Scale Factor",
                            f"{scale_factor:.2%}",
                            help=f"Your UAV mass relative to {benchmark.name}"
                        )
                        
                        # Visual comparison bars
                        st.markdown("**Visual Comparison:**")
                        
                        # Efficiency comparison bar
                        fig_eff = go.Figure()
                        fig_eff.add_trace(go.Bar(
                            x=['Your UAV', benchmark.name],
                            y=[uav_system_efficiency*100, benchmark.system_efficiency*100],
                            marker_color=[colors[0], colors[idx+1]],
                            text=[f'{uav_system_efficiency*100:.1f}%', f'{benchmark.system_efficiency*100:.1f}%'],
                            textposition='auto'
                        ))
                        fig_eff.update_layout(
                            title='System Efficiency Comparison',
                            yaxis_title='Efficiency (%)',
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig_eff, use_container_width=True)
                        
                        # Energy density comparison bar
                        fig_ed = go.Figure()
                        fig_ed.add_trace(go.Bar(
                            x=['Your UAV', benchmark.name],
                            y=[uav_energy_density_wh_per_kg, benchmark.energy_density_wh_per_kg],
                            marker_color=[colors[0], colors[idx+1]],
                            text=[f'{uav_energy_density_wh_per_kg:.0f}', f'{benchmark.energy_density_wh_per_kg:.0f}'],
                            textposition='auto'
                        ))
                        fig_ed.update_layout(
                            title='Energy Density Comparison (Wh/kg)',
                            yaxis_title='Energy Density (Wh/kg)',
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig_ed, use_container_width=True)
                    
                    # Delta analysis text
                    st.markdown("---")
                    st.markdown("### 📝 Detailed Delta Analysis")
                    st.markdown(delta_results['analysis_text'])
        
        else:
            st.info("👆 Click 'Compare with Industry Benchmarks' to see how your UAV compares against industry leaders.")
    
    with tab6:
        st.header("📊 Trade-Space Analysis")
        st.markdown("""
        **Parameter Trade-Space Visualization**
        
        This analysis explores the design trade-space between insulation thickness and cruise altitude,
        showing how these parameters affect total mission range. The contour plot reveals optimal
        design regions and helps identify the global optimum configuration.
        
        **Key Insights:**
        - **Insulation Thickness**: Thicker insulation reduces boil-off but increases tank weight
        - **Cruise Altitude**: Higher altitude improves L/D ratio but affects ambient temperature
        - **Global Optimum**: The point where range is maximized across the entire trade-space
        """)
        
        # Parameter ranges for trade-space analysis
        col_ts1, col_ts2 = st.columns(2)
        
        with col_ts1:
            st.subheader("📐 Parameter Ranges")
            insulation_min = st.slider(
                "Min Insulation Thickness (mm)",
                min_value=10.0,
                max_value=50.0,
                value=10.0,
                step=5.0,
                help="Minimum insulation thickness for trade-space analysis. Physics: Lower bound of MLI thickness range. Thinner → Higher BOG but lower tank weight."
            )
            insulation_max = st.slider(
                "Max Insulation Thickness (mm)",
                min_value=50.0,
                max_value=150.0,
                value=100.0,
                step=5.0,
                help="Maximum insulation thickness for trade-space analysis. Physics: Upper bound of MLI thickness range. Thicker → Lower BOG but higher tank weight."
            )
            n_insulation_points = st.slider(
                "Number of Insulation Points",
                min_value=20,
                max_value=50,
                value=30,
                step=5,
                help="Number of discrete insulation thickness values to evaluate. More points → Higher resolution contour plot → Longer computation time."
            )
        
        with col_ts2:
            st.subheader("✈️ Altitude Range")
            altitude_min = st.slider(
                "Min Cruise Altitude (m)",
                min_value=0,
                max_value=5000,
                value=0,
                step=500,
                help="Minimum cruise altitude for trade-space analysis. Physics: Lower altitude → Higher air density → Higher drag but warmer ambient (more BOG)."
            )
            altitude_max = st.slider(
                "Max Cruise Altitude (m)",
                min_value=5000,
                max_value=15000,
                value=12000,
                step=500,
                help="Maximum cruise altitude for trade-space analysis. Physics: Higher altitude → Lower air density (ρ = ρ₀exp(-h/H)) → Lower drag but colder ambient (less BOG)."
            )
            n_altitude_points = st.slider(
                "Number of Altitude Points",
                min_value=20,
                max_value=50,
                value=30,
                step=5,
                help="Number of discrete altitude values to evaluate. More points → Higher resolution contour plot → Longer computation time."
            )
        
        # System parameters (can be made adjustable)
        st.markdown("---")
        col_params1, col_params2 = st.columns(2)
        
        with col_params1:
            st.subheader("⚙️ System Parameters")
            fuel_mass_ts = st.slider(
                "Fuel Mass (kg)",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Fuel mass for trade-space analysis. Physics: More fuel → Higher energy (E = m × LHV) → Longer range but higher takeoff mass."
            )
            payload_mass_ts = st.slider(
                "Payload Mass (kg)",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                help="Payload mass for trade-space analysis. Physics: Higher payload → Higher total mass → Higher drag power (P = mgv/L/D) → Reduced range."
            )
        
        with col_params2:
            st.subheader("🔧 Performance Parameters")
            base_lift_to_drag_ts = st.slider(
                "Base L/D Ratio",
                min_value=10.0,
                max_value=20.0,
                value=15.0,
                step=0.5,
                help="Base lift-to-drag ratio. Physics: L/D = CL/CD. Higher L/D → Lower drag power (P = mgv/L/D) → Longer range. Typical UAV: 12-18."
            )
            cruise_velocity_ts = st.slider(
                "Cruise Velocity (m/s)",
                min_value=20.0,
                max_value=50.0,
                value=30.0,
                step=5.0,
                help="Cruise velocity for trade-space analysis. Physics: Drag power P ∝ v³. Higher velocity → Shorter mission time but much higher power → Reduced range."
            )
        
        # Generate trade-space data
        if st.button("🔄 Generate Trade-Space Analysis", type="primary", use_container_width=True):
            with st.spinner("Computing trade-space data..."):
                # Create parameter grids
                insulation_values = np.linspace(insulation_min, insulation_max, n_insulation_points)
                altitude_values = np.linspace(altitude_min, altitude_max, n_altitude_points)
                
                # Initialize range matrix
                range_matrix = np.zeros((len(altitude_values), len(insulation_values)))
                
                # Calculate range for each combination
                empty_mass_ts = 10.0  # kg
                total_efficiency_ts = 0.5
                tank_radius_ts = 0.3  # m
                tank_length_ts = 1.5  # m
                
                progress_bar = st.progress(0)
                total_iterations = len(altitude_values) * len(insulation_values)
                iteration = 0
                
                for i, altitude in enumerate(altitude_values):
                    for j, insulation in enumerate(insulation_values):
                        try:
                            range_matrix[i, j] = calculate_range_with_altitude(
                                insulation_thickness_mm=insulation,
                                cruise_altitude_m=altitude,
                                fuel_mass=fuel_mass_ts,
                                payload_mass=payload_mass_ts,
                                empty_mass=empty_mass_ts,
                                base_lift_to_drag=base_lift_to_drag_ts,
                                total_efficiency=total_efficiency_ts,
                                cruise_velocity=cruise_velocity_ts,
                                tank_radius=tank_radius_ts,
                                tank_length=tank_length_ts
                            )
                        except ValueError:
                            range_matrix[i, j] = 0.0  # Default fallback for invalid parameters
                        iteration += 1
                        progress_bar.progress(iteration / total_iterations)
                
                # Find global optimum
                max_range_idx = np.unravel_index(np.argmax(range_matrix), range_matrix.shape)
                opt_altitude_idx, opt_insulation_idx = max_range_idx
                opt_altitude = altitude_values[opt_altitude_idx]
                opt_insulation = insulation_values[opt_insulation_idx]
                opt_range = range_matrix[opt_altitude_idx, opt_insulation_idx]
                
                # Store in session state
                st.session_state['trade_space_data'] = {
                    'insulation_values': insulation_values,
                    'altitude_values': altitude_values,
                    'range_matrix': range_matrix,
                    'opt_altitude': opt_altitude,
                    'opt_insulation': opt_insulation,
                    'opt_range': opt_range
                }
        
        # Display trade-space plot if data exists
        if 'trade_space_data' in st.session_state:
            ts_data = st.session_state['trade_space_data']
            
            st.markdown("---")
            st.subheader("📈 Trade-Space Contour Plot")
            
            # Create publication-quality contour plot
            fig_trade = go.Figure()
            
            # Create contour plot with grayscale-friendly colors
            # Using a sequential grayscale colormap suitable for publications
            # Reversed 'Greys' (darker = higher values) for better contrast
            fig_trade.add_trace(go.Contour(
                x=ts_data['insulation_values'],
                y=ts_data['altitude_values'],
                z=ts_data['range_matrix'],
                colorscale='Greys_r',  # Reversed grayscale for publications (darker = higher)
                contours=dict(
                    start=np.min(ts_data['range_matrix']),
                    end=np.max(ts_data['range_matrix']),
                    size=(np.max(ts_data['range_matrix']) - np.min(ts_data['range_matrix'])) / 15,
                    showlines=True,
                    showlabels=True,
                    labelfont=dict(size=10, family='serif')
                ),
                colorbar=dict(
                    title=dict(
                        text='Range (km)',
                        font=dict(size=14, family='serif')
                    ),
                    tickfont=dict(size=12, family='serif')
                ),
                hovertemplate='Insulation: %{x:.1f} mm<br>Altitude: %{y:.0f} m<br>Range: %{z:.2f} km<extra></extra>'
            ))
            
            # Add global optimum marker (grayscale-friendly for publications)
            fig_trade.add_trace(go.Scatter(
                x=[ts_data['opt_insulation']],
                y=[ts_data['opt_altitude']],
                mode='markers+text',
                marker=dict(
                    size=25,
                    color='black',  # Black for grayscale visibility
                    symbol='star',
                    line=dict(width=3, color='white'),  # White border for contrast
                    opacity=0.9
                ),
                text=['★ Global Optimum'],
                textposition='top center',
                textfont=dict(size=13, family='serif', color='black'),
                name='Global Optimum',
                hovertemplate=f'<b>Global Optimum</b><br>Insulation: {ts_data["opt_insulation"]:.1f} mm<br>Altitude: {ts_data["opt_altitude"]:.0f} m<br>Range: {ts_data["opt_range"]:.2f} km<extra></extra>'
            ))
            
            # Publication-quality formatting
            fig_trade.update_layout(
                title=dict(
                    text=r'$\text{Trade-Space Analysis: Range vs. Insulation Thickness and Cruise Altitude}$',
                    x=0.5,
                    xanchor='center',
                    font=dict(size=16, family='serif')
                ),
                xaxis=dict(
                    title=dict(
                        text=r'$\text{Insulation Thickness } \delta_{\text{MLI}} \text{ (mm)}$',
                        font=dict(size=14, family='serif')
                    ),
                    tickfont=dict(size=12, family='serif'),
                    gridcolor='lightgray',
                    gridwidth=1,
                    showgrid=True
                ),
                yaxis=dict(
                    title=dict(
                        text=r'$\text{Cruise Altitude } h \text{ (m)}$',
                        font=dict(size=14, family='serif')
                    ),
                    tickfont=dict(size=12, family='serif'),
                    gridcolor='lightgray',
                    gridwidth=1,
                    showgrid=True
                ),
                width=900,
                height=700,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='serif'),
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    xanchor='left',
                    yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=11, family='serif')
                )
            )
            
            st.plotly_chart(fig_trade, use_container_width=True)
            
            # Display optimum results
            st.markdown("---")
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            
            with col_opt1:
                st.metric(
                    "Optimal Insulation Thickness",
                    f"{ts_data['opt_insulation']:.1f} mm",
                    help="Insulation thickness at global optimum"
                )
            
            with col_opt2:
                st.metric(
                    "Optimal Cruise Altitude",
                    f"{ts_data['opt_altitude']:.0f} m",
                    help="Cruise altitude at global optimum"
                )
            
            with col_opt3:
                st.metric(
                    "Maximum Range",
                    f"{ts_data['opt_range']:.2f} km",
                    help="Maximum achievable range at optimum point"
                )
            
            # Analysis insights
            st.markdown("---")
            st.subheader("💡 Trade-Space Insights")
            
            # Calculate range statistics
            range_min = np.min(ts_data['range_matrix'])
            range_max = np.max(ts_data['range_matrix'])
            range_mean = np.mean(ts_data['range_matrix'])
            range_std = np.std(ts_data['range_matrix'])
            
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                st.markdown("**Range Statistics:**")
                st.markdown(f"- Minimum Range: {range_min:.2f} km")
                st.markdown(f"- Maximum Range: {range_max:.2f} km")
                st.markdown(f"- Mean Range: {range_mean:.2f} km")
                st.markdown(f"- Standard Deviation: {range_std:.2f} km")
                st.markdown(f"- Range Improvement: {((range_max - range_min) / range_min * 100):.1f}%")
            
            with col_insight2:
                st.markdown("**Optimal Configuration Analysis:**")
                
                # Analyze why this is optimal
                insulation_ratio = ts_data['opt_insulation'] / insulation_max
                altitude_ratio = ts_data['opt_altitude'] / altitude_max
                
                if insulation_ratio < 0.3:
                    insulation_insight = "Low insulation suggests boil-off is acceptable at this altitude."
                elif insulation_ratio > 0.7:
                    insulation_insight = "High insulation indicates significant boil-off concerns."
                else:
                    insulation_insight = "Moderate insulation balances weight and thermal performance."
                
                if altitude_ratio < 0.3:
                    altitude_insight = "Low altitude suggests drag benefits don't outweigh temperature effects."
                elif altitude_ratio > 0.7:
                    altitude_insight = "High altitude maximizes L/D improvement from reduced air density."
                else:
                    altitude_insight = "Moderate altitude balances aerodynamic and thermal trade-offs."
                
                st.markdown(f"- **Insulation**: {insulation_insight}")
                st.markdown(f"- **Altitude**: {altitude_insight}")
                
                # Calculate sensitivity
                opt_idx_alt = np.argmin(np.abs(ts_data['altitude_values'] - ts_data['opt_altitude']))
                opt_idx_ins = np.argmin(np.abs(ts_data['insulation_values'] - ts_data['opt_insulation']))
                
                # Sensitivity to insulation (partial derivative approximation)
                if opt_idx_ins > 0 and opt_idx_ins < len(ts_data['insulation_values']) - 1:
                    d_range_d_insulation = (ts_data['range_matrix'][opt_idx_alt, opt_idx_ins + 1] - 
                                           ts_data['range_matrix'][opt_idx_alt, opt_idx_ins - 1]) / \
                                          (ts_data['insulation_values'][opt_idx_ins + 1] - 
                                           ts_data['insulation_values'][opt_idx_ins - 1])
                    st.markdown(f"- **Insulation Sensitivity**: {d_range_d_insulation:.3f} km/mm")
                
                # Sensitivity to altitude
                if opt_idx_alt > 0 and opt_idx_alt < len(ts_data['altitude_values']) - 1:
                    d_range_d_altitude = (ts_data['range_matrix'][opt_idx_alt + 1, opt_idx_ins] - 
                                         ts_data['range_matrix'][opt_idx_alt - 1, opt_idx_ins]) / \
                                        (ts_data['altitude_values'][opt_idx_alt + 1] - 
                                         ts_data['altitude_values'][opt_idx_alt - 1])
                    st.markdown(f"- **Altitude Sensitivity**: {d_range_d_altitude*1000:.3f} km/km")
            
            # Export option for publication
            st.markdown("---")
            st.subheader("📥 Export for Publication")
            st.markdown("""
            **Publication Guidelines:**
            - The contour plot uses LaTeX-formatted axis labels suitable for academic journals
            - Grayscale colormap ensures readability in black-and-white printing
            - High-resolution output (900×700 px) suitable for journal submission
            - Use the download button in the plot toolbar to export as PNG or SVG
            """)
        else:
            st.info("👆 Click 'Generate Trade-Space Analysis' to compute and visualize the parameter trade-space.")
    
    # Project Methodology Section
    st.markdown("---")
    with st.expander("📚 Project Methodology & Technical Approach", expanded=False):
        st.markdown("""
        ### **1D-3D Coupling Architecture**
        
        This simulation tool employs a sophisticated **1D-3D coupling approach** to bridge the gap between 
        simplified analytical models and high-fidelity computational fluid dynamics (CFD):
        
        **1D Analytical Models:**
        - **Modified Breguet Range Equation**: Extends classical range equation for electric propulsion systems
          - Accounts for variable efficiency chains (fuel cell → motor → propulsive)
          - Incorporates cryogenic tank weight penalties
          - Handles altitude-dependent air density effects
        
        - **Nernst Equation**: Models PEM fuel cell voltage-current characteristics
          - Open-circuit voltage: E_OCV = 1.23V (theoretical)
          - Activation losses: η_act = (RT/αF)ln(i/i₀)
          - Ohmic losses: η_ohm = iR_ohm
          - Concentration losses: η_conc = (RT/nF)ln(1 - i/i_L)
        
        - **Fourier's Law**: Calculates heat leak through Multi-Layer Insulation (MLI)
          - Q = kA(ΔT)/t where k is effective thermal conductivity
          - MLI effective k ≈ 0.0001-0.0005 W/m·K (highly dependent on layer count)
          - Boil-off rate: ṁ_BOG = Q / L_vaporization
        
        **3D CFD Integration:**
        - **Pressure Drop Lookup Tables**: Pre-computed CFD results for hydrogen flow through fuel cell stack
          - Maps mass flow rate → pressure drop (ΔP)
          - Accounts for flow distribution, manifold losses, and channel friction
          - Enables real-time pump power calculation: P_pump = ṁ·ΔP/ρ·η_pump
        
        - **Aerodynamic L/D Variation**: Altitude-dependent lift-to-drag ratio
          - Standard atmosphere model: ρ(h) = ρ₀exp(-h/H_scale)
          - L/D improves with altitude due to reduced air density
          - Incorporated into range calculations via modified Breguet equation
        
        **Coupling Strategy:**
        1. **1D Mission Loop**: Iterates through flight phases (takeoff, climb, cruise, descent)
        2. **3D Lookup Interpolation**: At each time step, queries CFD tables for pressure drop
        3. **Energy Balance**: Continuously tracks fuel consumption, BOG losses, and efficiency reductions
        4. **Physics Verification**: Energy conservation check (E_in = E_out + E_losses) to 4 decimal places
        
        ---
        
        ### **CoolProp Library Integration**
        
        The simulation leverages the **CoolProp** thermodynamic property library for accurate cryogenic 
        hydrogen properties:
        
        **Key Properties Accessed:**
        - **Density**: ρ(T,P) - Critical for tank sizing and mass calculations
          - Liquid hydrogen at 20K: ρ ≈ 70.8 kg/m³ (validated against NIST data)
        
        - **Latent Heat of Vaporization**: L_vap(T) - Essential for BOG calculations
          - At 20K: L_vap ≈ 448 kJ/kg
          - Temperature-dependent: L_vap decreases as temperature approaches critical point
        
        - **Specific Heat Capacity**: c_p(T,P) - Used in thermal analysis
          - Liquid phase: c_p ≈ 9.7 kJ/kg·K at 20K
          - Vapor phase: c_p ≈ 14.3 kJ/kg·K at 300K
        
        - **Thermal Conductivity**: k(T,P) - For heat transfer calculations
          - Liquid hydrogen: k ≈ 0.1 W/m·K at 20K
        
        **Implementation:**
        ```python
        from CoolProp.CoolProp import PropsSI
        density = PropsSI('D', 'T', temperature_K, 'P', pressure_Pa, 'ParaHydrogen')
        latent_heat = PropsSI('H', 'T', T_vapor, 'Q', 1, 'ParaHydrogen') - 
                      PropsSI('H', 'T', T_liquid, 'Q', 0, 'ParaHydrogen')
        ```
        
        **Benefits:**
        - **Accuracy**: NIST-validated property data ensures physically correct simulations
        - **Temperature Dependence**: Properties vary with operating conditions (not constant values)
        - **Real-time Calculation**: Properties computed on-the-fly based on current tank state
        - **Research-Grade**: Suitable for academic publication and engineering validation
        
        ---
        
        ### **Validation & Verification**
        
        **Physics Verification:**
        - Energy balance checked every simulation run: |E_in - (E_out + E_losses)| < 0.0001 J
        - Mass conservation: Initial fuel = Consumed + BOG + Remaining
        - Range validation: Cross-checked against analytical Breguet equation
        
        **Benchmark Comparison:**
        - Results validated against AeroDelft Phoenix (TU Delft) and Airbus ZEROe specifications
        - System efficiency and energy density metrics within expected ranges
        
        **Uncertainty Quantification:**
        - Monte Carlo analysis with Gaussian parameter distributions
        - P90 range calculation (industry standard safety metric)
        - Confidence intervals for mission planning
        
        ---
        
        **References:**
        - CoolProp: Bell et al. (2014), "Pure and Pseudo-pure Fluid Thermophysical Property Evaluation"
        - Breguet Range Equation: Breguet (1920), "Theorie des avions"
        - Nernst Equation: Nernst (1889), "Die elektromotorische Wirksamkeit der Ionen"
        - Fourier's Law: Fourier (1822), "Théorie analytique de la chaleur"
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p><strong>Hydrogen-Electric UAV Propulsion Simulator</strong></p>
    <p>Advanced Aerospace Propulsion Analysis Tool | Master's Research Portfolio</p>
    <p>Based on Modified Breguet Range Equation, Nernst Equation, and Fourier's Law</p>
    </p>1D-3D Coupling with CoolProp Integration | Validated Against Industry Benchmarks</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
