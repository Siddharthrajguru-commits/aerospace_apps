"""
Validation Tests for Hydrogen-Electric UAV Propulsion Simulator

This module validates the accuracy of physics and fuel cell calculations
against established standards and realistic operating ranges.

Author: Senior Aerospace Propulsion Engineer
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules with error handling
try:
    from core.physics import HydrogenProperties
    HAS_PHYSICS = True
except ImportError:
    HAS_PHYSICS = False
    HydrogenProperties = None

from core.fuel_cell import PEMFuelCell


def validate_lh2_density():
    """
    Validate LH2 density at 20K against NIST standard value.
    
    NIST Reference: Liquid hydrogen density at 20K (saturation pressure)
    Standard value: ~70.8 kg/m³
    
    Returns:
        tuple: (calculated_density, nist_value, percent_error, is_valid)
    """
    print("=" * 70)
    print("VALIDATION TEST 1: LH2 Density at 20K")
    print("=" * 70)
    
    if not HAS_PHYSICS:
        print("\nWARNING: SKIPPED - CoolProp not available")
        print("   Install CoolProp to run density validation:")
        print("   pip install CoolProp")
        return None, 70.8, None, False
    
    # NIST standard value for liquid hydrogen density at 20K
    nist_density = 70.8  # kg/m³
    
    # Initialize hydrogen properties
    try:
        props = HydrogenProperties()
        
        # Calculate density at 20K (saturation pressure)
        # At 20K, hydrogen is at saturation, so we need saturation pressure
        # For liquid hydrogen at 20K, pressure is approximately 0.1 MPa (100 kPa)
        temperature = 20.0  # K
        saturation_pressure = 101325  # Pa (atmospheric, approximate for validation)
        
        # Try to get density at saturation conditions
        # At 20K, hydrogen is liquid, so we use saturation pressure
        try:
            # Use saturation pressure for 20K
            # CoolProp can handle this with 'Q', 0 for saturated liquid
            import CoolProp.CoolProp as CP
            saturation_pressure_20k = CP.PropsSI('P', 'T', 20.0, 'Q', 0, 'Hydrogen')
            calculated_density = props.get_density(temperature, saturation_pressure_20k)
        except:
            # Fallback: use atmospheric pressure
            calculated_density = props.get_density(temperature, saturation_pressure)
        
        # Calculate percent error
        percent_error = abs((calculated_density - nist_density) / nist_density) * 100
        
        # Check if within 5% tolerance
        is_valid = percent_error <= 5.0
        
        print(f"NIST Standard Density:     {nist_density:.2f} kg/m³")
        print(f"Calculated Density:        {calculated_density:.2f} kg/m³")
        print(f"Percent Error:             {percent_error:.2f}%")
        print(f"Tolerance (5%):            {'PASS' if is_valid else 'FAIL'}")
        
        if not is_valid:
            print(f"\nWARNING: Density error ({percent_error:.2f}%) exceeds 5% tolerance!")
            print(f"   Expected: {nist_density:.2f} kg/m³")
            print(f"   Got:      {calculated_density:.2f} kg/m³")
        else:
            print(f"\nPASS: Density validation PASSED ({percent_error:.2f}% error)")
        
        return calculated_density, nist_density, percent_error, is_valid
        
    except Exception as e:
        print(f"\nERROR: Failed to calculate density: {e}")
        print("   This may be due to CoolProp not being installed.")
        print("   Install with: pip install CoolProp")
        return None, nist_density, None, False


def validate_polarization_curve():
    """
    Generate polarization curve and validate voltage stays within realistic range.
    
    Realistic voltage range for PEM fuel cells: 0.5V to 1.2V
    - Open circuit: ~1.2V
    - Maximum power: ~0.6-0.8V
    - Minimum (at high current): ~0.5V
    
    Returns:
        tuple: (voltages, current_densities, is_valid, min_voltage, max_voltage)
    """
    print("\n" + "=" * 70)
    print("VALIDATION TEST 2: Fuel Cell Polarization Curve")
    print("=" * 70)
    
    # Initialize fuel cell
    fuel_cell = PEMFuelCell(E_ocv=1.2, temperature=353.15)
    
    # Generate current density range
    # Typical PEMFC operates from 0 to ~2 A/m², but can go higher
    max_current_density = 3.0  # A/m²
    current_densities = np.linspace(0, max_current_density, 200)
    
    # Calculate cell voltages
    cell_voltages = fuel_cell.calculate_cell_voltage(current_densities)
    
    # Validate voltage range
    min_voltage = np.min(cell_voltages)
    max_voltage = np.max(cell_voltages)
    
    # Expected ranges
    expected_min = 0.5  # V (minimum realistic voltage)
    expected_max = 1.2  # V (open circuit voltage)
    
    # Check if voltages are within realistic range
    voltage_below_min = min_voltage < expected_min
    voltage_above_max = max_voltage > expected_max * 1.05  # Allow 5% above OCV
    
    is_valid = not voltage_below_min and not voltage_above_max
    
    print(f"Current Density Range:      0.0 to {max_current_density:.2f} A/m²")
    print(f"Calculated Voltage Range:  {min_voltage:.3f} to {max_voltage:.3f} V")
    print(f"Expected Voltage Range:    {expected_min:.2f} to {expected_max:.2f} V")
    print(f"Open Circuit Voltage:      {fuel_cell.E_ocv:.2f} V")
    
    # Check for issues
    warnings = []
    
    if voltage_below_min:
        error_percent = abs((min_voltage - expected_min) / expected_min) * 100
        if error_percent > 5.0:
            warnings.append(f"Minimum voltage ({min_voltage:.3f}V) is below realistic range ({expected_min}V)")
            print(f"\nWARNING: Minimum voltage ({min_voltage:.3f}V) is below realistic minimum ({expected_min}V)")
            print(f"   Error: {error_percent:.2f}%")
    
    if voltage_above_max * 1.05:
        error_percent = abs((max_voltage - expected_max) / expected_max) * 100
        if error_percent > 5.0:
            warnings.append(f"Maximum voltage ({max_voltage:.3f}V) exceeds realistic range ({expected_max}V)")
            print(f"\nWARNING: Maximum voltage ({max_voltage:.3f}V) exceeds realistic maximum ({expected_max}V)")
            print(f"   Error: {error_percent:.2f}%")
    
    # Check if OCV is correct (should be ~1.2V)
    ocv_error = abs(fuel_cell.E_ocv - expected_max) / expected_max * 100
    if ocv_error > 5.0:
        warnings.append(f"OCV ({fuel_cell.E_ocv:.3f}V) differs from expected ({expected_max}V)")
        print(f"\nWARNING: OCV ({fuel_cell.E_ocv:.3f}V) differs from expected ({expected_max}V)")
        print(f"   Error: {ocv_error:.2f}%")
    
    if is_valid and len(warnings) == 0:
        print(f"\nPASS: Polarization curve validation PASSED")
        print(f"  All voltages within realistic range (0.5V - 1.2V)")
    else:
        print(f"\nWARNING: Polarization curve validation has warnings")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Generate and save polarization curve plot
    try:
        plt.figure(figsize=(10, 7))
        
        # Main polarization curve
        plt.plot(current_densities, cell_voltages, 'b-', linewidth=2.5, 
                label='Cell Voltage')
        
        # Add voltage range boundaries
        plt.axhline(y=expected_max, color='g', linestyle='--', alpha=0.5, 
                   label=f'Expected Max ({expected_max}V)')
        plt.axhline(y=expected_min, color='r', linestyle='--', alpha=0.5, 
                   label=f'Expected Min ({expected_min}V)')
        
        # Highlight out-of-range regions
        if min_voltage < expected_min:
            plt.fill_between(current_densities, cell_voltages, expected_min, 
                            where=(cell_voltages < expected_min), 
                            alpha=0.3, color='red', label='Below Minimum')
        
        plt.xlabel('Current Density (A/m²)', fontsize=12, fontweight='bold')
        plt.ylabel('Cell Voltage (V)', fontsize=12, fontweight='bold')
        plt.title('PEM Fuel Cell Polarization Curve\nValidation Plot', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=10, loc='best')
        plt.xlim([0, max_current_density])
        plt.ylim([0, max_voltage * 1.1])
        
        # Add text box with validation results
        validation_text = f"Validation Results:\n"
        validation_text += f"Voltage Range: {min_voltage:.3f} - {max_voltage:.3f} V\n"
        validation_text += f"Status: {'PASS' if is_valid else 'WARNINGS'}"
        
        plt.text(0.02, 0.98, validation_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path(__file__).parent.parent
        plot_path = output_dir / 'polarization_curve_validation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPASS: Polarization curve plot saved: {plot_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"\nWARNING: Could not generate plot: {e}")
    
    return cell_voltages, current_densities, is_valid, min_voltage, max_voltage


def validate_voltage_at_specific_current():
    """
    Validate voltage at specific current densities to ensure realistic behavior.
    
    Checks voltage at:
    - 0 A/m² (should be ~1.2V OCV)
    - 1 A/m² (typical operating point, should be ~0.7-0.9V)
    - 2 A/m² (high current, should be ~0.5-0.7V)
    """
    print("\n" + "=" * 70)
    print("VALIDATION TEST 3: Voltage at Specific Current Densities")
    print("=" * 70)
    
    fuel_cell = PEMFuelCell(E_ocv=1.2, temperature=353.15)
    
    # Test points: (current_density, expected_voltage_range)
    test_points = [
        (0.0, (1.15, 1.25)),      # OCV: should be ~1.2V ± 0.05V
        (1.0, (0.65, 0.95)),      # Typical operating: 0.7-0.9V
        (2.0, (0.50, 0.75)),      # High current: 0.5-0.7V
    ]
    
    all_valid = True
    
    for current_density, (expected_min, expected_max) in test_points:
        voltage = fuel_cell.calculate_cell_voltage(current_density)
        
        is_in_range = expected_min <= voltage <= expected_max
        
        print(f"\nCurrent Density: {current_density:.1f} A/m²")
        print(f"  Calculated Voltage: {voltage:.3f} V")
        print(f"  Expected Range:    {expected_min:.2f} - {expected_max:.2f} V")
        print(f"  Status:            {'PASS' if is_in_range else 'OUT OF RANGE'}")
        
        if not is_in_range:
            if voltage < expected_min:
                error_percent = abs((voltage - expected_min) / expected_min) * 100
            else:
                error_percent = abs((voltage - expected_max) / expected_max) * 100
            
            if error_percent > 5.0:
                print(f"  WARNING: Voltage error ({error_percent:.2f}%) exceeds 5% tolerance!")
                all_valid = False
    
    if all_valid:
        print(f"\nPASS: All voltage checks PASSED")
    else:
        print(f"\nWARNING: Some voltage checks have warnings")
    
    return all_valid


def main():
    """
    Run all validation tests and generate summary report.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "VALIDATION TEST SUITE")
    print(" " * 10 + "Hydrogen-Electric UAV Propulsion Simulator")
    print("=" * 70)
    print()
    
    results = {}
    all_passed = True
    
    # Test 1: LH2 Density
    try:
        density_result = validate_lh2_density()
        results['density'] = density_result[3] if density_result[0] is not None else False
        if not results['density']:
            all_passed = False
    except Exception as e:
        print(f"\nERROR: Density validation failed: {e}")
        results['density'] = False
        all_passed = False
    
    # Test 2: Polarization Curve
    try:
        polarization_result = validate_polarization_curve()
        results['polarization'] = polarization_result[2]
        if not polarization_result[2]:
            all_passed = False
    except Exception as e:
        print(f"\nERROR: Polarization curve validation failed: {e}")
        results['polarization'] = False
        all_passed = False
    
    # Test 3: Voltage at Specific Currents
    try:
        voltage_result = validate_voltage_at_specific_current()
        results['voltage_points'] = voltage_result
        if not voltage_result:
            all_passed = False
    except Exception as e:
        print(f"\nERROR: Voltage point validation failed: {e}")
        results['voltage_points'] = False
        all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"LH2 Density Test:           {'PASS' if results.get('density', False) else 'WARNINGS'}")
    print(f"Polarization Curve Test:    {'PASS' if results.get('polarization', False) else 'WARNINGS'}")
    print(f"Voltage Points Test:       {'PASS' if results.get('voltage_points', False) else 'WARNINGS'}")
    print("=" * 70)
    
    if all_passed:
        print("\nSUCCESS: ALL VALIDATION TESTS PASSED")
        print("  All calculations are within acceptable tolerances.")
    else:
        print("\nWARNING: SOME VALIDATION TESTS HAVE WARNINGS")
        print("  Please review the warnings above.")
        print("  Values exceeding 5% tolerance may indicate:")
        print("    - Parameter tuning needed")
        print("    - Model limitations")
        print("    - Input condition differences")
    
    print("\n" + "=" * 70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
