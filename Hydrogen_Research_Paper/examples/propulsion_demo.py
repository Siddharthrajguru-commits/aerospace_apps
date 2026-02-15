"""
Demonstration of Propulsion System Integration with Fuel Cell Efficiency

This script demonstrates how CFD-based pressure drop analysis affects
fuel cell system efficiency through parasitic pump power losses.

Author: Senior Aerospace Propulsion Engineer
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from core.propulsion import PropulsionSystem
from core.fuel_cell import PEMFuelCell


def demonstrate_efficiency_reduction():
    """
    Demonstrate how pressure drop affects system efficiency.
    """
    print("=" * 70)
    print("Propulsion System - CFD-Based Pressure Drop Analysis")
    print("=" * 70)
    print()
    
    # Initialize propulsion system with lookup table
    lookup_path = Path(__file__).parent.parent / 'data' / 'injector_pressure_drop.csv'
    propulsion = PropulsionSystem(lookup_table_path=str(lookup_path))
    
    # Initialize fuel cell
    fuel_cell = PEMFuelCell(E_ocv=1.2, temperature=353.15)
    
    # Base efficiency (without pressure drop losses)
    base_efficiency = 0.5  # 50%
    
    # Fuel cell parameters
    fuel_cell_area = 0.1  # m² (fuel cell active area)
    current_density = 1000  # A/m² (typical operating point)
    
    # Calculate fuel cell power
    cell_voltage = fuel_cell.calculate_cell_voltage(current_density)
    fuel_cell_power = current_density * cell_voltage * fuel_cell_area  # W
    
    print(f"Fuel Cell Configuration:")
    print(f"  Active Area: {fuel_cell_area:.3f} m²")
    print(f"  Current Density: {current_density:.0f} A/m²")
    print(f"  Cell Voltage: {cell_voltage:.3f} V")
    print(f"  Fuel Cell Power: {fuel_cell_power:.1f} W")
    print()
    
    # Mass flow rate range (kg/s) - typical for small UAV
    mass_flow_rates = np.linspace(0.001, 0.01, 50)
    
    # Calculate efficiency reduction for each mass flow rate
    efficiencies_base = np.full_like(mass_flow_rates, base_efficiency)
    efficiencies_adjusted = []
    pump_powers = []
    pressure_drops = []
    
    for mfr in mass_flow_rates:
        # Get pressure drop from lookup table
        pressure_drop = propulsion.get_pressure_drop(mfr)
        pressure_drops.append(pressure_drop)
        
        # Calculate adjusted efficiency
        adj_eff = propulsion.get_efficiency_with_pressure_drop(
            base_efficiency, mfr, fuel_cell_power
        )
        efficiencies_adjusted.append(adj_eff)
        
        # Calculate pump power
        pump_power = propulsion.calculate_pump_power(mfr, pressure_drop)
        pump_powers.append(pump_power)
    
    efficiencies_adjusted = np.array(efficiencies_adjusted)
    pump_powers = np.array(pump_powers)
    pressure_drops = np.array(pressure_drops)
    
    # Print results at key points
    print("=" * 70)
    print("Efficiency Analysis Results")
    print("=" * 70)
    print(f"{'Mass Flow':<15} {'Pressure Drop':<18} {'Pump Power':<15} {'Efficiency':<15} {'Reduction':<15}")
    print(f"{'(kg/s)':<15} {'(kPa)':<18} {'(W)':<15} {'(adjusted)':<15} {'(%)':<15}")
    print("-" * 70)
    
    key_indices = [0, len(mass_flow_rates)//4, len(mass_flow_rates)//2, 
                   3*len(mass_flow_rates)//4, len(mass_flow_rates)-1]
    
    for idx in key_indices:
        mfr = mass_flow_rates[idx]
        pd_kpa = pressure_drops[idx] / 1000
        pp = pump_powers[idx]
        eff_adj = efficiencies_adjusted[idx]
        reduction = (1 - eff_adj/base_efficiency) * 100
        
        print(f"{mfr:<15.4f} {pd_kpa:<18.2f} {pp:<15.2f} {eff_adj:<15.4f} {reduction:<15.2f}")
    
    print()
    print(f"Base Efficiency (no pressure drop): {base_efficiency:.4f}")
    print(f"Min Adjusted Efficiency: {np.min(efficiencies_adjusted):.4f}")
    print(f"Max Efficiency Reduction: {(1 - np.min(efficiencies_adjusted)/base_efficiency)*100:.2f}%")
    print()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Pressure Drop vs Mass Flow Rate
    ax1 = axes[0, 0]
    ax1.plot(mass_flow_rates * 1000, pressure_drops / 1000, 'b-', linewidth=2)
    ax1.set_xlabel('Mass Flow Rate (g/s)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Pressure Drop (kPa)', fontsize=11, fontweight='bold')
    ax1.set_title('Injector Pressure Drop vs Mass Flow Rate\n(CFD Lookup Table)', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Pump Power vs Mass Flow Rate
    ax2 = axes[0, 1]
    ax2.plot(mass_flow_rates * 1000, pump_powers, 'r-', linewidth=2)
    ax2.set_xlabel('Mass Flow Rate (g/s)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Pump Power (W)', fontsize=11, fontweight='bold')
    ax2.set_title('Parasitic Pump Power vs Mass Flow Rate', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency Reduction
    ax3 = axes[1, 0]
    ax3.plot(mass_flow_rates * 1000, efficiencies_base * 100, 'g--', 
            linewidth=2, label='Base Efficiency (no losses)')
    ax3.plot(mass_flow_rates * 1000, efficiencies_adjusted * 100, 'b-', 
            linewidth=2, label='Adjusted Efficiency (with pressure drop)')
    ax3.set_xlabel('Mass Flow Rate (g/s)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('System Efficiency (%)', fontsize=11, fontweight='bold')
    ax3.set_title('System Efficiency: Base vs Adjusted\n(Accounting for Pump Losses)', 
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency Reduction Percentage
    ax4 = axes[1, 1]
    efficiency_reduction_pct = (1 - efficiencies_adjusted / base_efficiency) * 100
    ax4.plot(mass_flow_rates * 1000, efficiency_reduction_pct, 'm-', linewidth=2)
    ax4.fill_between(mass_flow_rates * 1000, 0, efficiency_reduction_pct, 
                     alpha=0.3, color='magenta')
    ax4.set_xlabel('Mass Flow Rate (g/s)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Efficiency Reduction (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Efficiency Reduction Due to Pressure Drop\n(Parasitic Pump Power)', 
                 fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(__file__).parent.parent / 'propulsion_efficiency_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {output_path}")
    
    plt.show()
    
    print()
    print("=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print("1. Pressure drop increases quadratically with mass flow rate")
    print("2. Pump power increases proportionally with pressure drop")
    print("3. System efficiency decreases as mass flow rate increases")
    print("4. At high flow rates, parasitic losses can reduce efficiency by 5-15%")
    print("5. This CFD-based analysis provides realistic efficiency predictions")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_efficiency_reduction()
