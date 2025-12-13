# CubeSat Debris Removal - Hohmann Transfer Simulation

A Python simulation of a CubeSat performing a Hohmann transfer maneuver to intercept space debris in Low Earth Orbit (LEO). This project demonstrates orbital mechanics calculations and 3D visualization for a portfolio application to TU Delft's MSc Space Engineering program.

## Overview

This simulation models a **Chaser** satellite (CubeSat) performing a rendezvous maneuver to intercept a piece of space debris using a standard two-burn Hohmann transfer:

- **Initial Orbit**: Circular LEO at 400 km altitude
- **Target Orbit**: Circular LEO at 800 km altitude (coplanar)
- **Maneuver**: Two-burn Hohmann transfer

## Features

- ✅ Accurate orbital mechanics calculations using `poliastro`
- ✅ Delta-V (Δv) calculations for both burns
- ✅ Time of Flight (TOF) calculation
- ✅ Interactive 3D visualization of orbits
- ✅ Production-ready, PEP8 compliant code
- ✅ Comprehensive documentation and comments
- ✅ Interactive Dash web application with real-time updates
- ✅ Physics breakdown with step-by-step calculations

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Dash Web Application (Recommended)

Run the interactive dashboard:

```bash
python app.py
```

The app will start a local web server (usually at `http://127.0.0.1:8050`) where you can:
- Enter custom altitude values
- See real-time 3D visualization updates
- View detailed physics calculations
- Explore mission parameters interactively

### Standalone Scripts

Run individual simulation scripts:

```bash
# Original matplotlib version
python debris_simulation.py

# Interactive Plotly version
python debris_simulation_interactive.py

# Static dashboard version
python dashboard_simulation.py
```

## Output

The simulation prints:
- **First Burn Delta-V** (periapsis): Velocity change required to enter transfer orbit
- **Second Burn Delta-V** (apoapsis): Velocity change required to circularize at target altitude
- **Total Delta-V**: Total fuel cost for the mission
- **Time of Flight**: Duration of the transfer maneuver
- **Start Location**: 3D coordinates at first burn
- **Intercept Location**: 3D coordinates at rendezvous

The 3D visualization shows:
- 🌍 Earth (reference body)
- 🟢 Initial Orbit (Green) - Chaser satellite starting position
- 🔴 Target Orbit (Red) - Debris location
- 🟡 Transfer Orbit (Yellow) - Hohmann transfer path
- 🔵 Start Point (Cyan) - First burn location
- 🟣 Intercept Point (Magenta) - Second burn location

## Physics Background

### Hohmann Transfer

A Hohmann transfer is an elliptical orbit used to transfer between two circular orbits in the same plane. It consists of:

1. **First Burn (Periapsis)**: Increases velocity to enter the elliptical transfer orbit
2. **Coast Phase**: Spacecraft follows the transfer ellipse for half an orbit
3. **Second Burn (Apoapsis)**: Increases velocity again to circularize at the target altitude

### Key Equations

- **Semi-major axis**: `a = (r₁ + r₂) / 2`
- **Eccentricity**: `e = (r₂ - r₁) / (r₂ + r₁)`
- **Vis-viva equation**: `v = √(μ(2/r - 1/a))`
- **Orbital period**: `T = 2π√(a³/μ)`

Where:
- `r₁` = initial orbit radius
- `r₂` = target orbit radius
- `μ` = gravitational parameter (GM_earth)

## Project Structure

```
CubeSat_(V1)/
├── app.py                              # Main Dash web application (Interactive)
├── debris_simulation.py                # Original matplotlib version
├── debris_simulation_interactive.py    # Interactive Plotly version
├── dashboard_simulation.py             # Static dashboard version
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── .gitignore                          # Git ignore rules
```

## Dependencies

- `poliastro` - Orbital mechanics calculations
- `astropy` - Physical units and constants
- `matplotlib` - Plotting and visualization
- `numpy` - Numerical computations
- `plotly` - Interactive 3D visualization
- `dash` - Web application framework
- `dash-bootstrap-components` - UI components

## Author

Aerospace Engineering Undergraduate  
-Siddharth Rajguru 

## License

This project is provided as-is for educational and portfolio purposes.

