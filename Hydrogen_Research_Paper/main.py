"""
Main Mission Simulation Script for Hydrogen-Electric UAV Propulsion Simulator

This script runs a complete mission simulation for a hydrogen-electric UAV
with specified payload and fuel mass, outputting range calculations and
visualization plots.

Author: Senior Aerospace Propulsion Engineer
"""

import numpy as np
import matplotlib.pyplot as plt
from core.mission import MissionProfile
from core.tank import LH2Tank


def main():
    """
    Execute mission simulation for hydrogen-electric UAV.
    
    Mission parameters:
    - Payload: 5 kg
    - LH2 Fuel: 2 kg
    - Output: Total range in kilometers and fuel consumption plot
    """
    # Mission parameters
    payload_mass = 5.0  # kg
    fuel_mass = 2.0  # kg (LH2)
    empty_mass = 10.0  # kg (aircraft structure + systems)
    
    # Initialize mission profile
    mission = MissionProfile(
        payload_mass=payload_mass,
        fuel_mass=fuel_mass,
        empty_mass=empty_mass,
        lift_to_drag=15.0,  # Typical for efficient UAV
        total_efficiency=0.5,  # 50% total system efficiency
        hydrogen_energy_density=120e6  # J/kg (120 MJ/kg)
    )
    
    # Initialize tank for gravimetric index calculation
    tank = LH2Tank(tank_radius=0.3, tank_length=1.5, ambient_temp=293.15)
    
    # Calculate gravimetric index
    gi_results = tank.calculate_gravimetric_index(
        fuel_mass=fuel_mass,
        payload_mass=payload_mass,
        empty_aircraft_mass=empty_mass
    )
    
    print("=" * 60)
    print("Hydrogen-Electric UAV Propulsion Simulator")
    print("Mission Simulation Results")
    print("=" * 60)
    print(f"\nMission Parameters:")
    print(f"  Payload Mass: {payload_mass} kg")
    print(f"  Fuel Mass (LH2): {fuel_mass} kg")
    print(f"  Empty Aircraft Mass: {empty_mass} kg")
    print(f"  Start Mass: {mission.calculate_start_mass():.2f} kg")
    print(f"  End Mass: {mission.calculate_end_mass():.2f} kg")
    
    print(f"\nGravimetric Index Analysis:")
    print(f"  Gravimetric Index: {gi_results['gravimetric_index']:.3f}")
    print(f"  Target GI: {gi_results['target_gravimetric_index']:.3f}")
    print(f"  Tank Structure Mass: {gi_results['tank_structure_mass_kg']:.2f} kg")
    print(f"  Total System Mass: {gi_results['total_system_mass_kg']:.2f} kg")
    print(f"  Meets Target: {'Yes' if gi_results['meets_target'] else 'No'}")
    
    # Run mission simulation
    print(f"\nRunning Mission Simulation...")
    mission_results = mission.run_mission()
    
    # Extract cruise data for plotting
    cruise_data = mission_results['cruise_data']
    distances = [point['distance_km'] for point in cruise_data]
    fuel_remaining = [point['remaining_fuel_kg'] for point in cruise_data]
    
    # Add takeoff and climb phases to the plot data
    # Start from takeoff
    plot_distances = [0.0]
    plot_fuel = [fuel_mass]
    
    # Add takeoff point
    plot_distances.append(mission_results['takeoff']['distance_km'])
    plot_fuel.append(mission_results['takeoff']['remaining_fuel_kg'])
    
    # Add climb endpoint
    climb_end_dist = mission_results['takeoff']['distance_km'] + mission_results['climb']['distance_km']
    plot_distances.append(climb_end_dist)
    plot_fuel.append(mission_results['climb']['remaining_fuel_kg'])
    
    # Add cruise data
    plot_distances.extend(distances)
    plot_fuel.extend(fuel_remaining)
    
    # Add descent endpoint
    descent_end_dist = mission_results['total_range_km']
    plot_distances.append(descent_end_dist)
    plot_fuel.append(mission_results['descent']['remaining_fuel_kg'])
    
    # Print mission results
    print(f"\nMission Phase Summary:")
    print(f"  Takeoff Distance: {mission_results['takeoff']['distance_km']:.2f} km")
    print(f"  Climb Distance: {mission_results['climb']['distance_km']:.2f} km")
    print(f"  Cruise Distance: {mission_results['cruise_distance']:.2f} km")
    print(f"  Descent Distance: {mission_results['descent']['distance_km']:.2f} km")
    print(f"\n  Total Range: {mission_results['total_range_km']:.2f} km")
    print(f"  Total Fuel Consumed: {mission_results['total_fuel_consumed_kg']:.3f} kg")
    print(f"  Remaining Fuel: {mission_results['remaining_fuel_kg']:.3f} kg")
    
    # Calculate range using Breguet equation for comparison
    breguet_range = mission.calculate_range_breguet()
    print(f"\n  Theoretical Breguet Range: {breguet_range/1000:.2f} km")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(plot_distances, plot_fuel, 'b-', linewidth=2, label='Remaining Fuel')
    plt.xlabel('Distance Flown (km)', fontsize=12)
    plt.ylabel('Remaining Fuel (kg)', fontsize=12)
    plt.title('Hydrogen-Electric UAV Mission Profile\nRemaining Fuel vs. Distance Flown', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add phase markers
    plt.axvline(x=mission_results['takeoff']['distance_km'], color='g', 
                linestyle='--', alpha=0.5, label='Takeoff End')
    plt.axvline(x=climb_end_dist, color='orange', 
                linestyle='--', alpha=0.5, label='Climb End')
    plt.axvline(x=mission_results['total_range_km'] - mission_results['descent']['distance_km'], 
                color='r', linestyle='--', alpha=0.5, label='Descent Start')
    
    # Add text annotation for total range
    plt.text(0.02, 0.98, f'Total Range: {mission_results["total_range_km"]:.2f} km',
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('mission_profile.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'mission_profile.png'")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Mission Simulation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
