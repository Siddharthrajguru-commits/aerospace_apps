"""
Physics Verification Module

Provides energy balance verification and physics consistency checks for the
hydrogen-electric propulsion system simulation.

Author: Senior Aerospace Propulsion Engineer
"""

import numpy as np
import sys
import os
from typing import Dict, Any, Optional

# Check if running in Streamlit environment
try:
    import streamlit as st
    IN_STREAMLIT = True
except ImportError:
    IN_STREAMLIT = False


class PhysicsVerification:
    """
    Physics verification and energy balance checking for propulsion system.
    
    Verifies energy conservation: E_in = E_out + E_loss
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize physics verification.
        
        Args:
            verbose (bool): Print verification logs to terminal. Default True.
                            Automatically disabled in Streamlit environment.
        """
        # Disable verbose printing in Streamlit to avoid encoding issues
        self.verbose = verbose and not IN_STREAMLIT
        self.verification_log = []
    
    def verify_energy_balance(self, 
                              fuel_mass_kg: float,
                              hydrogen_energy_density_j_per_kg: float,
                              fuel_cell_output_energy_j: float,
                              motor_output_energy_j: float,
                              propulsive_energy_j: float,
                              fuel_cell_losses_j: float,
                              motor_losses_j: float,
                              propulsive_losses_j: float,
                              bog_losses_j: float,
                              simulation_name: str = "Simulation") -> Dict[str, Any]:
        """
        Verify energy balance: E_in = E_out + E_loss
        
        Args:
            fuel_mass_kg: Initial fuel mass in kg
            hydrogen_energy_density_j_per_kg: Hydrogen energy density in J/kg
            fuel_cell_output_energy_j: Energy output from fuel cell in J
            motor_output_energy_j: Energy output from motor in J
            propulsive_energy_j: Useful propulsive energy in J
            fuel_cell_losses_j: Fuel cell heat losses in J
            motor_losses_j: Motor/inverter losses in J
            propulsive_losses_j: Aerodynamic/propulsive losses in J
            bog_losses_j: Boil-off gas energy losses in J
            simulation_name: Name identifier for this simulation
            
        Returns:
            dict: Verification results with balance check and error
        """
        # Calculate total energy input
        E_in = fuel_mass_kg * hydrogen_energy_density_j_per_kg
        
        # Calculate total energy output (useful work)
        E_out = propulsive_energy_j
        
        # Calculate total losses
        E_loss = (fuel_cell_losses_j + motor_losses_j + 
                 propulsive_losses_j + bog_losses_j)
        
        # Total energy accounted for
        E_accounted = E_out + E_loss
        
        # Energy balance error
        E_error = E_in - E_accounted
        
        # Relative error (percentage)
        relative_error_pct = (E_error / E_in * 100) if E_in > 0 else 0.0
        
        # Verification result
        is_balanced = abs(relative_error_pct) < 0.01  # Within 0.01% tolerance
        
        result = {
            'simulation_name': simulation_name,
            'E_in': E_in,
            'E_out': E_out,
            'E_loss': E_loss,
            'E_accounted': E_accounted,
            'E_error': E_error,
            'relative_error_pct': relative_error_pct,
            'is_balanced': is_balanced,
            'fuel_cell_losses': fuel_cell_losses_j,
            'motor_losses': motor_losses_j,
            'propulsive_losses': propulsive_losses_j,
            'bog_losses': bog_losses_j
        }
        
        # Log verification
        if self.verbose:
            self._print_verification(result)
        
        # Store in log
        self.verification_log.append(result)
        
        return result
    
    def _print_verification(self, result: Dict[str, Any]):
        """Print energy balance verification to terminal."""
        # Skip printing in Streamlit environment to avoid encoding issues
        if IN_STREAMLIT:
            return
        
        # Skip printing if stdout/stderr are not available or have encoding issues
        try:
            # Test if we can write to stderr
            sys.stderr.write("")
        except (OSError, AttributeError):
            # If stderr is not available, skip printing entirely
            return
        
        try:
            # Use only ASCII-safe characters for maximum compatibility
            # Format values first to avoid issues in f-strings
            try:
                sim_name = str(result['simulation_name'])
                e_in = f"{result['E_in']:>20.4f}"
                e_out = f"{result['E_out']:>20.4f}"
                e_loss = f"{result['E_loss']:>20.4f}"
                fc_loss = f"{result['fuel_cell_losses']:>20.4f}"
                motor_loss = f"{result['motor_losses']:>20.4f}"
                prop_loss = f"{result['propulsive_losses']:>20.4f}"
                bog_loss = f"{result['bog_losses']:>20.4f}"
                e_accounted = f"{result['E_accounted']:>20.4f}"
                e_error = f"{result['E_error']:>20.4f}"
                rel_error = f"{result['relative_error_pct']:>19.4f}"
            except (OSError, ValueError, KeyError):
                # If formatting fails, skip printing
                return
            
            # Build output lines with simple string concatenation (no f-strings to avoid issues)
            output_lines = [
                "\n" + "="*80,
                "PHYSICS VERIFICATION: " + sim_name,
                "="*80,
                "Energy Balance: E_in = E_out + E_loss",
                "-"*80,
                "E_in  (Total Fuel Energy)     = " + e_in + " J",
                "E_out (Useful Propulsive)     = " + e_out + " J",
                "E_loss (Total Losses)         = " + e_loss + " J",
                "  - Fuel Cell Losses          = " + fc_loss + " J",
                "  - Motor/Inverter Losses     = " + motor_loss + " J",
                "  - Propulsive Losses         = " + prop_loss + " J",
                "  - BOG Losses                 = " + bog_loss + " J",
                "-"*80,
                "E_accounted (E_out + E_loss)  = " + e_accounted + " J",
                "E_error (E_in - E_accounted)  = " + e_error + " J",
                "Relative Error                = " + rel_error + " %",
                "-"*80,
            ]
            
            try:
                if result['is_balanced']:
                    output_lines.append("[OK] ENERGY BALANCE VERIFIED (within 0.01% tolerance)")
                else:
                    error_pct = f"{result['relative_error_pct']:.4f}"
                    output_lines.append("[WARNING] ENERGY BALANCE ERROR: " + error_pct + "% imbalance")
            except (OSError, ValueError, KeyError):
                output_lines.append("[INFO] Energy balance verification completed")
            
            output_lines.append("="*80 + "\n")
            
            # Write to stderr with comprehensive error handling
            for line in output_lines:
                try:
                    sys.stderr.write(line + "\n")
                    sys.stderr.flush()
                except (UnicodeEncodeError, OSError, AttributeError, ValueError):
                    # If stderr fails, try stdout
                    try:
                        sys.stdout.write(line + "\n")
                        sys.stdout.flush()
                    except (UnicodeEncodeError, OSError, AttributeError, ValueError):
                        # If both fail, skip this line and continue
                        continue
                        
        except (OSError, UnicodeEncodeError, AttributeError, ValueError, KeyError):
            # Silently fail if printing is not possible
            # This prevents crashes in environments where stdout/stderr are not available
            pass
        except Exception:
            # Catch any other unexpected errors
            pass
    
    def get_verification_summary(self) -> Dict[str, Any]:
        """
        Get summary of all verification results.
        
        Returns:
            dict: Summary statistics
        """
        if not self.verification_log:
            return {'total_simulations': 0}
        
        errors = [r['relative_error_pct'] for r in self.verification_log]
        balanced = [r['is_balanced'] for r in self.verification_log]
        
        return {
            'total_simulations': len(self.verification_log),
            'balanced_count': sum(balanced),
            'unbalanced_count': sum(not b for b in balanced),
            'mean_error_pct': np.mean(errors),
            'max_error_pct': np.max(np.abs(errors)),
            'min_error_pct': np.min(np.abs(errors))
        }
