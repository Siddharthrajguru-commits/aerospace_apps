# Propulsion System Module - Usage Guide

## Overview

The `PropulsionSystem` class integrates CFD-based fluid dynamics analysis into the hydrogen-electric propulsion simulator. It accounts for real-world pressure drops in injector systems and calculates the resulting parasitic pump power that reduces overall system efficiency.

## Key Features

1. **CFD Lookup Table Integration**: Reads pressure drop data from CSV files
2. **Parasitic Power Calculation**: Calculates pump power required to overcome pressure drops
3. **Efficiency Reduction**: Adjusts system efficiency based on pump losses
4. **Real-World Fluid Dynamics**: Incorporates actual injector characteristics

## Basic Usage

```python
from core.propulsion import PropulsionSystem
from core.fuel_cell import PEMFuelCell

# Initialize propulsion system (loads CSV lookup table)
propulsion = PropulsionSystem()

# Or specify custom lookup table path
propulsion = PropulsionSystem(lookup_table_path='path/to/your/data.csv')

# Get pressure drop for a given mass flow rate
mass_flow_rate = 0.005  # kg/s
pressure_drop = propulsion.get_pressure_drop(mass_flow_rate)
print(f"Pressure drop: {pressure_drop/1000:.2f} kPa")

# Calculate pump power
pump_power = propulsion.calculate_pump_power(mass_flow_rate)
print(f"Pump power: {pump_power:.2f} W")

# Calculate efficiency reduction
fuel_cell_power = 500  # W
base_efficiency = 0.5  # 50%

adjusted_efficiency = propulsion.get_efficiency_with_pressure_drop(
    base_efficiency, mass_flow_rate, fuel_cell_power
)
print(f"Base efficiency: {base_efficiency:.4f}")
print(f"Adjusted efficiency: {adjusted_efficiency:.4f}")
```

## CSV Lookup Table Format

The CSV file should contain two columns:

```csv
mass_flow_rate_kg_per_s,pressure_drop_Pa
0.001,75000
0.002,150000
0.005,495000
...
```

- `mass_flow_rate_kg_per_s`: Mass flow rate in kg/s
- `pressure_drop_Pa`: Pressure drop in Pascals (Pa)

## Integration with Mission Profile

To integrate pressure drop effects into mission calculations:

```python
from core.propulsion import PropulsionSystem
from core.mission import MissionProfile

# Initialize systems
propulsion = PropulsionSystem()
mission = MissionProfile(total_efficiency=0.5)

# During mission simulation, adjust efficiency based on mass flow rate
mass_flow_rate = 0.005  # kg/s (current fuel consumption rate)
fuel_cell_power = 500  # W (current power output)

# Get adjusted efficiency
adjusted_eff = propulsion.get_efficiency_with_pressure_drop(
    mission.total_efficiency, mass_flow_rate, fuel_cell_power
)

# Use adjusted efficiency in range calculations
# (modify mission.total_efficiency or use in Breguet equation)
```

## Physics Background

### Pressure Drop Calculation

Pressure drop across injectors follows the Darcy-Weisbach equation:

\[
\Delta P = f \frac{L}{D} \frac{\rho v^2}{2}
\]

For injector systems, this is typically simplified to:

\[
\Delta P = K \frac{\dot{m}^2}{2 \rho A^2}
\]

where:
- \(K\): Loss coefficient (typically 2-5 for aerospace injectors)
- \(\dot{m}\): Mass flow rate (kg/s)
- \(\rho\): Fluid density (kg/m³)
- \(A\): Injector cross-sectional area (m²)

### Pump Power

Parasitic pump power is calculated as:

\[
P_{pump} = \frac{\dot{m} \Delta P}{\rho \eta_{pump}}
\]

where \(\eta_{pump}\) is the combined pump and motor efficiency.

### Efficiency Reduction

The effective system efficiency accounting for pump losses:

\[
\eta_{effective} = \eta_{base} \frac{P_{fuel\_cell}}{P_{fuel\_cell} + P_{pump}}
\]

## Example: Efficiency Impact

At typical operating conditions:
- Mass flow rate: 0.005 kg/s
- Fuel cell power: 500 W
- Base efficiency: 50%

Result:
- Pressure drop: ~495 kPa
- Pump power: ~56 W
- Adjusted efficiency: ~44.4%
- **Efficiency reduction: ~11%**

This demonstrates the significant impact of parasitic losses on system performance.

## Running the Demo

```bash
python examples/propulsion_demo.py
```

This generates plots showing:
1. Pressure drop vs mass flow rate
2. Pump power vs mass flow rate
3. Efficiency reduction
4. Comparison of base vs adjusted efficiency
