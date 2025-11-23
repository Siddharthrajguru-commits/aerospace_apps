"""
CubeSat Debris Removal - Hohmann Transfer Simulation

This script simulates a Chaser satellite performing a rendezvous maneuver
to intercept a piece of space debris using a standard two-burn Hohmann transfer.

Author: Aerospace Engineering Undergraduate
Project: Portfolio for TU Delft MSc Space Engineering Application
"""

import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.constants import GM_earth
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_hohmann_transfer(initial_altitude, target_altitude):
    """
    Calculate Hohmann transfer orbit parameters and Delta-V requirements.
    
    Parameters
    ----------
    initial_altitude : float
        Initial circular orbit altitude in kilometers
    target_altitude : float
        Target circular orbit altitude in kilometers
    
    Returns
    -------
    dict
        Dictionary containing transfer orbit parameters and Delta-V values
    """
    # Earth's radius (poliastro uses standard value)
    R_earth = Earth.R.to(u.km).value
    
    # Calculate orbital radii
    r1 = R_earth + initial_altitude  # Initial orbit radius (km)
    r2 = R_earth + target_altitude   # Target orbit radius (km)
    
    # Convert to astropy units
    r1 = r1 * u.km
    r2 = r2 * u.km
    
    # Gravitational parameter (km^3/s^2)
    mu = GM_earth.to(u.km**3 / u.s**2)
    
    # Circular orbit velocities
    v1_circular = np.sqrt(mu / r1)  # Initial circular orbit velocity
    v2_circular = np.sqrt(mu / r2)  # Target circular orbit velocity
    
    # Semi-major axis of transfer ellipse
    a_transfer = (r1 + r2) / 2
    
    # Velocities at transfer orbit periapsis and apoapsis
    v1_transfer_peri = np.sqrt(mu * (2 / r1 - 1 / a_transfer))  # At periapsis
    v2_transfer_apo = np.sqrt(mu * (2 / r2 - 1 / a_transfer))   # At apoapsis
    
    # Delta-V calculations
    delta_v1 = v1_transfer_peri - v1_circular  # First burn (periapsis)
    delta_v2 = v2_circular - v2_transfer_apo    # Second burn (apoapsis)
    total_delta_v = delta_v1 + delta_v2
    
    # Time of flight (half period of transfer ellipse)
    period_transfer = 2 * np.pi * np.sqrt(a_transfer**3 / mu)
    time_of_flight = period_transfer / 2
    
    return {
        'r1': r1,
        'r2': r2,
        'a_transfer': a_transfer,
        'delta_v1': delta_v1,
        'delta_v2': delta_v2,
        'total_delta_v': total_delta_v,
        'time_of_flight': time_of_flight,
        'v1_circular': v1_circular,
        'v2_circular': v2_circular,
        'v1_transfer_peri': v1_transfer_peri,
        'v2_transfer_apo': v2_transfer_apo
    }


def create_orbits(initial_altitude, target_altitude):
    """
    Create Orbit objects for initial, target, and transfer orbits.
    
    All orbits are coplanar (same orbital plane) for a standard Hohmann transfer.
    
    Parameters
    ----------
    initial_altitude : float
        Initial circular orbit altitude in kilometers
    target_altitude : float
        Target circular orbit altitude in kilometers
    
    Returns
    -------
    tuple
        (initial_orbit, target_orbit, transfer_orbit)
    """
    # Calculate transfer parameters
    params = calculate_hohmann_transfer(initial_altitude, target_altitude)
    
    # Earth's radius
    R_earth = Earth.R
    
    # Orbital radii
    r1 = R_earth + initial_altitude * u.km
    r2 = R_earth + target_altitude * u.km
    
    # Create coplanar circular orbits using classical orbital elements
    # All orbits share the same orbital plane (equatorial, coplanar)
    # Inclination = 0, RAAN = 0, Argument of periapsis = 0
    inc = 0 * u.deg      # Inclination (equatorial)
    raan = 0 * u.deg     # Right Ascension of Ascending Node
    argp = 0 * u.deg     # Argument of periapsis
    nu = 0 * u.deg       # True anomaly (start at reference point)
    
    # Initial circular orbit (400 km altitude)
    # For circular orbit: semi-major axis = radius, eccentricity = 0
    initial_orbit = Orbit.from_classical(
        Earth,
        a=r1,
        ecc=0 * u.one,
        inc=inc,
        raan=raan,
        argp=argp,
        nu=nu
    )
    
    # Target circular orbit (800 km altitude)
    target_orbit = Orbit.from_classical(
        Earth,
        a=r2,
        ecc=0 * u.one,
        inc=inc,
        raan=raan,
        argp=argp,
        nu=nu
    )
    
    # Transfer orbit: elliptical orbit from r1 (periapsis) to r2 (apoapsis)
    # Semi-major axis of transfer ellipse
    a_transfer = params['a_transfer']
    
    # Eccentricity of transfer ellipse: e = (r_apo - r_peri) / (r_apo + r_peri)
    e_transfer = (r2 - r1) / (r2 + r1)
    
    # Create transfer orbit - starts at periapsis (r1)
    transfer_orbit = Orbit.from_classical(
        Earth,
        a=a_transfer,
        ecc=e_transfer * u.one,
        inc=inc,
        raan=raan,
        argp=argp,
        nu=0 * u.deg  # Start at periapsis (true anomaly = 0)
    )
    
    return initial_orbit, target_orbit, transfer_orbit


def print_results(params):
    """
    Print mission parameters and Delta-V requirements.
    
    Parameters
    ----------
    params : dict
        Dictionary containing transfer orbit parameters and Delta-V values
    """
    print("\n" + "="*60)
    print("CUBESAT DEBRIS REMOVAL - HOHMANN TRANSFER ANALYSIS")
    print("="*60)
    print(f"\nInitial Orbit Altitude:    400 km (LEO)")
    print(f"Target Orbit Altitude:     800 km (LEO)")
    print(f"\nTransfer Orbit Semi-major Axis: {params['a_transfer']:.2f}")
    print(f"\n{'='*60}")
    print("DELTA-V REQUIREMENTS")
    print(f"{'='*60}")
    print(f"First Burn (Periapsis):    {params['delta_v1']:.4f}")
    print(f"Second Burn (Apoapsis):    {params['delta_v2']:.4f}")
    print(f"{'-'*60}")
    print(f"Total Delta-V:             {params['total_delta_v']:.4f}")
    print(f"\n{'='*60}")
    print("MISSION TIMELINE")
    print(f"{'='*60}")
    print(f"Time of Flight:            {params['time_of_flight']:.2f}")
    print(f"{'='*60}\n")


def visualize_orbits(initial_orbit, target_orbit, transfer_orbit):
    """
    Create an interactive 3D visualization of the orbits.
    
    The visualization shows:
    - Earth (rendered as a sphere)
    - Initial orbit in Green (Chaser satellite)
    - Target orbit in Red (Debris)
    - Transfer orbit in Yellow (Hohmann transfer path)
    
    Parameters
    ----------
    initial_orbit : Orbit
        Initial circular orbit object
    target_orbit : Orbit
        Target circular orbit object
    transfer_orbit : Orbit
        Transfer elliptical orbit object
    """
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Earth's radius for scaling
    R_earth = Earth.R.to(u.km).value
    
    # Sample points along each orbit for visualization
    # Generate true anomaly values from 0 to 2π
    nu_samples = np.linspace(0, 2 * np.pi, 200) * u.rad
    
    def plot_orbit(orbit, color, label, linestyle='-'):
        """Helper function to plot an orbit in 3D."""
        # Sample the orbit at different true anomalies
        positions = []
        for nu in nu_samples:
            # Create orbit at this true anomaly
            orbit_at_nu = Orbit.from_classical(
                Earth,
                a=orbit.a,
                ecc=orbit.ecc,
                inc=orbit.inc,
                raan=orbit.raan,
                argp=orbit.argp,
                nu=nu
            )
            # Get position vector (in km)
            r = orbit_at_nu.r.to(u.km).value
            positions.append(r)
        
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                color=color, label=label, linestyle=linestyle, linewidth=2)
    
    # Plot Earth as a sphere
    u_sphere = np.linspace(0, 2 * np.pi, 50)
    v_sphere = np.linspace(0, np.pi, 50)
    x_sphere = R_earth * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_sphere = R_earth * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_sphere = R_earth * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='blue')
    
    # Plot orbits
    plot_orbit(initial_orbit, "#00FF00", "Initial Orbit (Chaser)", '-')
    plot_orbit(target_orbit, "#FF0000", "Target Orbit (Debris)", '-')
    plot_orbit(transfer_orbit, "#FFFF00", "Transfer Orbit (Hohmann)", '--')
    
    # Set equal aspect ratio and labels
    max_range = max([initial_orbit.a.to(u.km).value, 
                     target_orbit.a.to(u.km).value]) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel('X (km)', fontsize=10)
    ax.set_ylabel('Y (km)', fontsize=10)
    ax.set_zlabel('Z (km)', fontsize=10)
    ax.set_title('CubeSat Debris Removal - Hohmann Transfer', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Display the interactive 3D plot
    # Users can rotate, zoom, and pan to examine the orbits
    plt.show()


def main():
    """
    Main function to execute the CubeSat debris removal simulation.
    """
    # Mission parameters
    INITIAL_ALTITUDE = 400  # km - Chaser satellite initial altitude
    TARGET_ALTITUDE = 800   # km - Debris target altitude
    
    print("\nInitializing CubeSat Debris Removal Simulation...")
    print("Calculating Hohmann Transfer Parameters...")
    
    # Calculate transfer parameters
    params = calculate_hohmann_transfer(INITIAL_ALTITUDE, TARGET_ALTITUDE)
    
    # Print results
    print_results(params)
    
    # Create orbit objects
    print("Generating orbit objects...")
    initial_orbit, target_orbit, transfer_orbit = create_orbits(
        INITIAL_ALTITUDE, TARGET_ALTITUDE
    )
    
    # Visualize orbits
    print("Creating 3D visualization...")
    visualize_orbits(initial_orbit, target_orbit, transfer_orbit)
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()

