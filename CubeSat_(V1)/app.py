"""
CubeSat Debris Removal - Interactive Dash Web Application

This Dash web application provides a fully interactive interface for visualizing
a Chaser satellite performing a rendezvous maneuver to intercept space debris
using a standard two-burn Hohmann transfer.

Features:
- Real-time altitude input (Chaser and Debris)
- Live 3D orbital simulation updates
- Interactive mission analysis table
- Dark space-themed interface
- Starfield background and high-resolution Earth
- Physics breakdown with step-by-step calculations

Author: Aerospace Engineering Undergraduate
Project: Portfolio for TU Delft MSc Space Engineering Application
"""

import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.constants import GM_earth
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc


# Initialize Dash app with dark theme and Font Awesome icons
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ]
)
app.title = "CubeSat Debris Removal - Hohmann Transfer"


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
        'initial_altitude': initial_altitude,
        'target_altitude': target_altitude,
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
    
    # Initial circular orbit
    initial_orbit = Orbit.from_classical(
        Earth,
        a=r1,
        ecc=0 * u.one,
        inc=inc,
        raan=raan,
        argp=argp,
        nu=nu
    )
    
    # Target circular orbit
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
    a_transfer = params['a_transfer']
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


def generate_starfield(num_stars=500, min_radius=20000):
    """
    Generate random 3D points far away from Earth to simulate stars.
    
    Parameters
    ----------
    num_stars : int
        Number of stars to generate
    min_radius : float
        Minimum distance from origin (km) for stars
    
    Returns
    -------
    tuple
        (x, y, z) arrays of star positions
    """
    # Generate random spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi, num_stars)  # Azimuthal angle
    phi = np.arccos(np.random.uniform(-1, 1, num_stars))  # Polar angle
    
    # Random distances beyond minimum radius
    max_radius = min_radius * 2
    r = np.random.uniform(min_radius, max_radius, num_stars)
    
    # Convert to Cartesian coordinates
    x_stars = r * np.sin(phi) * np.cos(theta)
    y_stars = r * np.sin(phi) * np.sin(theta)
    z_stars = r * np.cos(phi)
    
    return x_stars, y_stars, z_stars


def create_earth_sphere(R_earth, num_points=100):
    """
    Create a high-resolution 3D sphere mesh for Earth.
    
    Parameters
    ----------
    R_earth : float
        Earth's radius in kilometers
    num_points : int
        Number of points for sphere resolution
    
    Returns
    -------
    tuple
        (x, y, z) meshgrid arrays for sphere surface
    """
    # Create spherical coordinates
    u_sphere = np.linspace(0, 2 * np.pi, num_points)
    v_sphere = np.linspace(0, np.pi, num_points)
    
    # Create meshgrid
    u_mesh, v_mesh = np.meshgrid(u_sphere, v_sphere)
    
    # Convert to Cartesian coordinates
    x_sphere = R_earth * np.sin(v_mesh) * np.cos(u_mesh)
    y_sphere = R_earth * np.sin(v_mesh) * np.sin(u_mesh)
    z_sphere = R_earth * np.cos(v_mesh)
    
    return x_sphere, y_sphere, z_sphere


def sample_orbit_trajectory(orbit, num_samples):
    """
    Sample an orbit at different true anomalies to get trajectory points.
    
    Parameters
    ----------
    orbit : Orbit
        Orbit object to sample
    num_samples : int
        Number of sample points
    
    Returns
    -------
    tuple
        (x, y, z) arrays of position coordinates
    """
    positions = []
    # Sample true anomaly from 0 to 2π
    nu_samples = np.linspace(0, 2 * np.pi, num_samples) * u.rad
    
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
    return positions[:, 0], positions[:, 1], positions[:, 2]


# Generate starfield once (static background) - Reduced for performance
x_stars, y_stars, z_stars = generate_starfield(num_stars=200, min_radius=20000)

# Create Earth sphere once (static) - Reduced resolution for performance
R_earth = Earth.R.to(u.km).value
x_earth, y_earth, z_earth = create_earth_sphere(R_earth, num_points=50)


# Define app layout
app.layout = dbc.Container([
    # Professional Header with Gradient
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2(
                    "CubeSat Debris Removal",
                    className="mb-2",
                    style={
                        'color': '#FFFFFF',
                        'fontWeight': '700',
                        'fontSize': '32px',
                        'letterSpacing': '0.5px',
                        'textShadow': '2px 2px 4px rgba(0,0,0,0.5)',
                        'marginBottom': '5px'
                    }
                ),
                html.P(
                    "Hohmann Transfer Simulation & Mission Analysis",
                    className="mb-0",
                    style={
                        'color': '#B0BEC5',
                        'fontSize': '16px',
                        'fontWeight': '300',
                        'letterSpacing': '0.3px'
                    }
                )
            ], style={
                'background': 'linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%)',
                'padding': '25px 30px',
                'borderRadius': '10px',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.3)',
                'marginBottom': '25px',
                'border': '1px solid rgba(255,255,255,0.1)'
            })
        ], width=12)
    ]),
    
    # Professional Input Controls Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label([
                                html.I(className="fas fa-satellite", style={'marginRight': '8px', 'color': '#3498DB'}),
                                " Chaser Altitude (km)"
                            ], style={
                                'color': '#E0E0E0',
                                'fontSize': '15px',
                                'fontWeight': '600',
                                'marginBottom': '8px',
                                'display': 'block'
                            }),
                            dcc.Input(
                                id='chaser-altitude',
                                type='number',
                                value=400,
                                style={
                                    'width': '100%',
                                    'padding': '12px 15px',
                                    'fontSize': '15px',
                                    'backgroundColor': '#263238',
                                    'color': '#FFFFFF',
                                    'border': '2px solid #3498DB',
                                    'borderRadius': '8px',
                                    'transition': 'all 0.3s ease',
                                    'boxShadow': '0 2px 8px rgba(52, 152, 219, 0.2)'
                                }
                            )
                        ], width={'size': 5, 'offset': 1}),
                        dbc.Col([
                            html.Label([
                                html.I(className="fas fa-space-shuttle", style={'marginRight': '8px', 'color': '#E74C3C'}),
                                " Debris Altitude (km)"
                            ], style={
                                'color': '#E0E0E0',
                                'fontSize': '15px',
                                'fontWeight': '600',
                                'marginBottom': '8px',
                                'display': 'block'
                            }),
                            dcc.Input(
                                id='debris-altitude',
                                type='number',
                                value=800,
                                style={
                                    'width': '100%',
                                    'padding': '12px 15px',
                                    'fontSize': '15px',
                                    'backgroundColor': '#263238',
                                    'color': '#FFFFFF',
                                    'border': '2px solid #E74C3C',
                                    'borderRadius': '8px',
                                    'transition': 'all 0.3s ease',
                                    'boxShadow': '0 2px 8px rgba(231, 76, 60, 0.2)'
                                }
                            )
                        ], width=5)
                    ], className="g-3", align="center")
                ], style={'padding': '20px'})
            ], style={
                'backgroundColor': '#1e1e1e',
                'border': '1px solid #37474F',
                'borderRadius': '12px',
                'boxShadow': '0 4px 20px rgba(0,0,0,0.4)',
                'marginBottom': '20px'
            })
        ], width=12)
    ]),
    
    # Main Grid (Top Row) - No Gutters
    dbc.Row([
        # Left Column (Width = 9)
        dbc.Col([
            # Graph: 3D Visualization
            dcc.Graph(id='orbital-visualization', style={'height': '65vh'}),
            
            # Professional Legend with styled badges
            html.Div([
                html.Div([
                    html.Span([
                        html.Span(style={
                            'display': 'inline-block',
                            'width': '14px',
                            'height': '14px',
                            'backgroundColor': '#00FF88',
                            'borderRadius': '3px',
                            'marginRight': '8px',
                            'verticalAlign': 'middle',
                            'boxShadow': '0 0 8px rgba(0,255,136,0.5)'
                        }),
                        html.Span("Chaser Orbit", style={'color': '#E0E0E0', 'fontSize': '13px', 'fontWeight': '500'})
                    ], style={'display': 'inline-block', 'marginRight': '15px', 'marginLeft': '5px', 'padding': '8px 12px', 'backgroundColor': 'rgba(0,255,136,0.1)', 'borderRadius': '6px', 'border': '1px solid rgba(0,255,136,0.3)', 'whiteSpace': 'nowrap'}),
                    
                    html.Span([
                        html.Span(style={
                            'display': 'inline-block',
                            'width': '14px',
                            'height': '14px',
                            'backgroundColor': '#FF4444',
                            'borderRadius': '3px',
                            'marginRight': '8px',
                            'verticalAlign': 'middle',
                            'boxShadow': '0 0 8px rgba(255,68,68,0.5)'
                        }),
                        html.Span("Debris Orbit", style={'color': '#E0E0E0', 'fontSize': '13px', 'fontWeight': '500'})
                    ], style={'display': 'inline-block', 'marginRight': '15px', 'padding': '8px 12px', 'backgroundColor': 'rgba(255,68,68,0.1)', 'borderRadius': '6px', 'border': '1px solid rgba(255,68,68,0.3)', 'whiteSpace': 'nowrap'}),
                    
                    html.Span([
                        html.Span(style={
                            'display': 'inline-block',
                            'width': '20px',
                            'height': '0px',
                            'borderTop': '2px dashed #FFD700',
                            'marginRight': '8px',
                            'verticalAlign': 'middle',
                            'marginTop': '5px',
                            'boxShadow': '0 0 8px rgba(255,215,0,0.5)'
                        }),
                        html.Span("Transfer", style={'color': '#E0E0E0', 'fontSize': '13px', 'fontWeight': '500'})
                    ], style={'display': 'inline-block', 'marginRight': '15px', 'padding': '8px 12px', 'backgroundColor': 'rgba(255,215,0,0.1)', 'borderRadius': '6px', 'border': '1px solid rgba(255,215,0,0.3)', 'whiteSpace': 'nowrap'}),
                    
                    html.Span([
                        html.Span(style={
                            'display': 'inline-block',
                            'width': '10px',
                            'height': '10px',
                            'backgroundColor': '#00FFFF',
                            'borderRadius': '50%',
                            'marginRight': '8px',
                            'verticalAlign': 'middle',
                            'boxShadow': '0 0 8px rgba(0,255,255,0.8)'
                        }),
                        html.Span("Start", style={'color': '#E0E0E0', 'fontSize': '13px', 'fontWeight': '500'})
                    ], style={'display': 'inline-block', 'marginRight': '15px', 'padding': '8px 12px', 'backgroundColor': 'rgba(0,255,255,0.1)', 'borderRadius': '6px', 'border': '1px solid rgba(0,255,255,0.3)', 'whiteSpace': 'nowrap'}),
                    
                    html.Span([
                        html.Span(style={
                            'display': 'inline-block',
                            'width': '10px',
                            'height': '10px',
                            'backgroundColor': '#FF00FF',
                            'borderRadius': '50%',
                            'marginRight': '8px',
                            'verticalAlign': 'middle',
                            'boxShadow': '0 0 8px rgba(255,0,255,0.8)'
                        }),
                        html.Span("Intercept", style={'color': '#E0E0E0', 'fontSize': '13px', 'fontWeight': '500'})
                    ], style={'display': 'inline-block', 'marginRight': '5px', 'padding': '8px 12px', 'backgroundColor': 'rgba(255,0,255,0.1)', 'borderRadius': '6px', 'border': '1px solid rgba(255,0,255,0.3)', 'whiteSpace': 'nowrap'})
                ], style={
                    'textAlign': 'center',
                    'padding': '12px 8px',
                    'backgroundColor': 'rgba(0,0,0,0.4)',
                    'borderRadius': '10px',
                    'border': '1px solid rgba(255,255,255,0.1)',
                    'backdropFilter': 'blur(10px)',
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'justifyContent': 'center',
                    'gap': '10px'
                })
            ], style={'marginTop': '15px', 'marginBottom': '10px'})
        ], width=9),
        
        # Right Column (Width = 3)
        dbc.Col([
            # Mission Analysis Table
            html.Div(id='mission-table')
        ], width=3)
    ], className="g-0"),
    
    # Bottom Row (Calculations) - Full Width, ~20vh height
    dbc.Row([
        dbc.Col([
            html.Div(id='physics-breakdown', style={'height': '20vh', 'overflowY': 'auto'})
        ], width=12)
    ], className="g-0", style={'marginTop': '10px'}),
    
], fluid=True, style={
    'background': 'linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 50%, #0a0a0a 100%)',
    'minHeight': '100vh',
    'padding': '25px',
    'fontFamily': "'Segoe UI', 'Roboto', 'Arial', sans-serif"
})


# Callback to update the dashboard, table, and physics breakdown
@app.callback(
    [Output('orbital-visualization', 'figure'),
     Output('mission-table', 'children'),
     Output('physics-breakdown', 'children')],
    [Input('chaser-altitude', 'value'),
     Input('debris-altitude', 'value')]
)
def update_dashboard(chaser_alt, debris_alt):
    """
    Update the dashboard when altitude inputs change.
    
    Parameters
    ----------
    chaser_alt : float
        Chaser satellite altitude in km
    debris_alt : float
        Debris altitude in km
    
    Returns
    -------
    tuple
        (go.Figure, html.Div, html.Div) - Updated Plotly figure, mission table, and physics breakdown content
    """
    # Validate inputs
    if chaser_alt is None or debris_alt is None:
        chaser_alt = 400
        debris_alt = 800
    
    if chaser_alt <= 0:
        chaser_alt = 400
    if debris_alt <= 0:
        debris_alt = 800
    
    # Calculate transfer parameters
    params = calculate_hohmann_transfer(chaser_alt, debris_alt)
    
    # Create orbit objects
    orb_chaser, orb_debris, orb_transfer = create_orbits(chaser_alt, debris_alt)
    
    # Calculate 3D coordinates at key moments
    # Start Point (t=0): Chaser location at first burn (periapsis of transfer orbit)
    # The transfer orbit starts at periapsis (nu=0), which is at the initial orbit radius
    start_r = orb_chaser.r.to(u.km).value  # Position vector at start
    start_x = round(start_r[0], 1)
    start_y = round(start_r[1], 1)
    start_z = round(start_r[2], 1)
    
    # Intercept Point (t_final): Location when rendezvous happens (apoapsis of transfer orbit)
    # Create transfer orbit at apoapsis (nu=π) to get intercept coordinates
    intercept_orbit = Orbit.from_classical(
        Earth,
        a=orb_transfer.a,
        ecc=orb_transfer.ecc,
        inc=orb_transfer.inc,
        raan=orb_transfer.raan,
        argp=orb_transfer.argp,
        nu=180 * u.deg  # Apoapsis (true anomaly = π)
    )
    intercept_r = intercept_orbit.r.to(u.km).value  # Position vector at intercept
    intercept_x = round(intercept_r[0], 1)
    intercept_y = round(intercept_r[1], 1)
    intercept_z = round(intercept_r[2], 1)
    
    # Create single 3D scene figure (no subplots)
    fig = go.Figure()
    
    # Add starfield to 3D scene
    fig.add_trace(
        go.Scatter3d(
            x=x_stars,
            y=y_stars,
            z=z_stars,
            mode='markers',
            marker=dict(
                size=1.5,
                color='white',
                opacity=0.6
            ),
            name='Starfield',
            showlegend=False,
            hovertemplate='Star<br>Distance: %{r:.0f} km<extra></extra>'
        )
    )
    
    # Add Earth as a 3D surface
    fig.add_trace(
        go.Surface(
            x=x_earth,
            y=y_earth,
            z=z_earth,
            colorscale=[[0, '#1E3A8A'], [0.3, '#2563EB'], [0.5, '#3B82F6'], 
                       [0.7, '#60A5FA'], [0.85, '#22C55E'], [1, '#16A34A']],
            showscale=False,
            name='Earth',
            lighting=dict(
                ambient=0.4,
                diffuse=0.9,
                specular=0.3,
                roughness=0.4,
                fresnel=0.2
            ),
            lightposition=dict(x=10000, y=10000, z=10000),
            hovertemplate='Earth<br>Radius: 6371 km<extra></extra>'
        )
    )
    
    # Sample orbit trajectories - Reduced for performance
    num_points = 150
    
    # Sample chaser orbit (initial orbit) - Green line
    chaser_x, chaser_y, chaser_z = sample_orbit_trajectory(orb_chaser, num_points)
    
    # Sample debris orbit (target orbit) - Red line
    debris_x, debris_y, debris_z = sample_orbit_trajectory(orb_debris, num_points)
    
    # Sample transfer orbit - Yellow dashed line
    transfer_x, transfer_y, transfer_z = sample_orbit_trajectory(orb_transfer, num_points)
    
    # Add Chaser orbit (Green)
    fig.add_trace(
        go.Scatter3d(
            x=chaser_x,
            y=chaser_y,
            z=chaser_z,
            mode='lines',
            line=dict(
                color='#00FF88',  # Professional green
                width=5
            ),
            name='Chaser Orbit',
            showlegend=False,
            hovertemplate='Chaser Orbit<br>X: %{x:.0f} km<br>Y: %{y:.0f} km<br>Z: %{z:.0f} km<extra></extra>'
        )
    )
    
    # Add Debris orbit (Red)
    fig.add_trace(
        go.Scatter3d(
            x=debris_x,
            y=debris_y,
            z=debris_z,
            mode='lines',
            line=dict(
                color='#FF4444',  # Professional red
                width=5
            ),
            name='Debris Orbit',
            showlegend=False,
            hovertemplate='Debris Orbit<br>X: %{x:.0f} km<br>Y: %{y:.0f} km<br>Z: %{z:.0f} km<extra></extra>'
        )
    )
    
    # Add Transfer orbit (Yellow dashed)
    fig.add_trace(
        go.Scatter3d(
            x=transfer_x,
            y=transfer_y,
            z=transfer_z,
            mode='lines',
            line=dict(
                color='#FFD700',  # Professional gold
                width=5,
                dash='dash'  # Dashed line
            ),
            name='Transfer Orbit (Hohmann)',
            showlegend=False,
            hovertemplate='Transfer Orbit<br>X: %{x:.0f} km<br>Y: %{y:.0f} km<br>Z: %{z:.0f} km<extra></extra>'
        )
    )
    
    # Add Start Point marker (Cyan sphere) - Periapsis of transfer orbit
    fig.add_trace(
        go.Scatter3d(
            x=[start_x],
            y=[start_y],
            z=[start_z],
            mode='markers',
            marker=dict(
                size=8,
                color='#00FFFF',
                symbol='circle',
                line=dict(color='white', width=2),
                opacity=0.9
            ),
            name='Start Point (Burn 1)',
            showlegend=False,
            hovertemplate='Start Point (Burn 1)<br>X: %{x:.1f} km<br>Y: %{y:.1f} km<br>Z: %{z:.1f} km<extra></extra>'
        )
    )
    
    # Add Intercept Point marker (Magenta sphere) - Apoapsis of transfer orbit
    fig.add_trace(
        go.Scatter3d(
            x=[intercept_x],
            y=[intercept_y],
            z=[intercept_z],
            mode='markers',
            marker=dict(
                size=8,
                color='#FF00FF',
                symbol='circle',
                line=dict(color='white', width=2),
                opacity=0.9
            ),
            name='Intercept Point (Burn 2)',
            showlegend=False,
            hovertemplate='Intercept Point (Burn 2)<br>X: %{x:.1f} km<br>Y: %{y:.1f} km<br>Z: %{z:.1f} km<extra></extra>'
        )
    )
    
    # Calculate appropriate axis limits
    max_range = max([
        np.max(np.abs(chaser_x)), np.max(np.abs(chaser_y)), np.max(np.abs(chaser_z)),
        np.max(np.abs(debris_x)), np.max(np.abs(debris_y)), np.max(np.abs(debris_z)),
        np.max(np.abs(transfer_x)), np.max(np.abs(transfer_y)), np.max(np.abs(transfer_z))
    ]) * 1.1
    
    # Configure 3D scene (left column) with visible axes and gridlines
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text='X (km)', font=dict(size=12)),
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.3)',
                gridwidth=1,
                showbackground=True,
                backgroundcolor='rgba(0, 0, 0, 0.1)',
                showline=True,
                linecolor='rgba(150, 150, 150, 0.8)',
                linewidth=2,
                zeroline=True,
                zerolinecolor='rgba(100, 100, 100, 0.5)',
                zerolinewidth=1,
                range=[-max_range, max_range]
            ),
            yaxis=dict(
                title=dict(text='Y (km)', font=dict(size=12)),
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.3)',
                gridwidth=1,
                showbackground=True,
                backgroundcolor='rgba(0, 0, 0, 0.1)',
                showline=True,
                linecolor='rgba(150, 150, 150, 0.8)',
                linewidth=2,
                zeroline=True,
                zerolinecolor='rgba(100, 100, 100, 0.5)',
                zerolinewidth=1,
                range=[-max_range, max_range]
            ),
            zaxis=dict(
                title=dict(text='Z (km)', font=dict(size=12)),
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.3)',
                gridwidth=1,
                showbackground=True,
                backgroundcolor='rgba(0, 0, 0, 0.1)',
                showline=True,
                linecolor='rgba(150, 150, 150, 0.8)',
                linewidth=2,
                zeroline=True,
                zerolinecolor='rgba(100, 100, 100, 0.5)',
                zerolinewidth=1,
                range=[-max_range, max_range]
            ),
            aspectmode='cube'
        ),
        template='plotly_dark',
        height=800,
        showlegend=False  # Disable default legend - using custom legend in right column
    )
    
    # Create mission analysis table data
    chaser_alt = f"{params['initial_altitude']:.0f} km"
    debris_alt = f"{params['target_altitude']:.0f} km"
    delta_v1 = f"{params['delta_v1'].to(u.km/u.s).value:.4f} km/s"
    delta_v2 = f"{params['delta_v2'].to(u.km/u.s).value:.4f} km/s"
    total_dv = f"{params['total_delta_v'].to(u.km/u.s).value:.4f} km/s"
    tof_seconds = params['time_of_flight'].to(u.s).value
    tof_minutes = tof_seconds / 60
    time_of_flight = f"{tof_minutes:.2f} min ({tof_seconds:.0f} s)"
    
    # Format coordinate strings
    start_location = f"({start_x}, {start_y}, {start_z}) km"
    intercept_location = f"({intercept_x}, {intercept_y}, {intercept_z}) km"
    
    # Professional Mission Analysis Table
    mission_table = dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.I(className="fas fa-table", style={'marginRight': '10px', 'color': '#3498DB'}),
                html.Span("Mission Analysis", style={'color': '#FFFFFF', 'fontWeight': '700', 'fontSize': '16px', 'letterSpacing': '0.5px'})
            ])
        ], style={
            'background': 'linear-gradient(135deg, #2C3E50 0%, #34495E 100%)',
            'borderBottom': '2px solid #3498DB',
            'padding': '12px 15px',
            'borderRadius': '10px 10px 0 0'
        }),
        dbc.CardBody([
            html.Div([
                dcc.Graph(
                    figure=go.Figure(data=[
                        go.Table(
                            header=dict(
                                values=['<b>Metric</b>', '<b>Value</b>'],
                                fill_color='#34495E',  # Solid color instead of gradient
                                align='left',
                                font=dict(size=13, color='white', family='Arial, sans-serif'),
                                height=24,
                                line=dict(color='#3498DB', width=2)
                            ),
                            cells=dict(
                                values=[
                                    ['Chaser Altitude', 'Debris Altitude', 'Delta-V 1', 'Delta-V 2', 'Total Delta-V', 'Time of Flight', 'Start Location (x, y, z)', 'Intercept Location (x, y, z)'],
                                    [chaser_alt, debris_alt, delta_v1, delta_v2, total_dv, time_of_flight, start_location, intercept_location]
                                ],
                                fill_color=['#263238', '#1e1e1e', '#263238', '#1e1e1e', '#263238', '#1e1e1e', '#263238', '#1e1e1e'],  # Alternating row colors
                                align='left',
                                font=dict(size=12, color='#E0E0E0', family='Arial, sans-serif'),
                                height=22,
                                line=dict(color='#37474F', width=1)
                            )
                        )
                    ]),
                    config={'displayModeBar': False},
                    style={
                        'height': 'auto',
                        'overflowX': 'hidden',
                        'overflowY': 'hidden'
                    }
                )
            ], style={
                'overflowX': 'hidden',
                'overflowY': 'hidden',
                'width': '100%',
                'height': 'auto'
            })
        ], style={'padding': '10px', 'overflowX': 'hidden', 'overflowY': 'hidden', 'backgroundColor': '#1e1e1e'})
    ], style={
        'backgroundColor': '#1e1e1e',
        'border': '1px solid #37474F',
        'borderRadius': '12px',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.4)',
        'height': '100%',
        'overflowX': 'hidden',
        'overflowY': 'hidden'
    })
    
    # Create Physics Breakdown content
    # Extract values from params (already calculated)
    r1_km = params['r1'].to(u.km).value
    r2_km = params['r2'].to(u.km).value
    a_transfer_km = params['a_transfer'].to(u.km).value
    v_initial = params['v1_circular'].to(u.km/u.s).value
    v_transfer_p = params['v1_transfer_peri'].to(u.km/u.s).value
    v_transfer_a = params['v2_transfer_apo'].to(u.km/u.s).value
    v_final = params['v2_circular'].to(u.km/u.s).value
    delta_v1_val = params['delta_v1'].to(u.km/u.s).value
    delta_v2_val = params['delta_v2'].to(u.km/u.s).value
    
    # Professional Physics Breakdown - Horizontal CardGroup layout
    physics_content = dbc.CardGroup([
        # Card 1: Orbital Parameters
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-circle-notch", style={'marginRight': '8px', 'color': '#3498DB'}),
                html.Span("Orbital Parameters", style={'color': 'white', 'fontWeight': '700', 'fontSize': '13px'})
            ], style={
                'background': 'linear-gradient(135deg, #34495E 0%, #2C3E50 100%)',
                'padding': '10px 12px',
                'borderBottom': '2px solid #3498DB'
            }),
            dbc.CardBody([
                html.P([
                    html.Span("r₁: ", style={'color': '#B0BEC5', 'fontSize': '11px'}),
                    html.Span(f"{r1_km:.2f} km", style={'color': '#E0E0E0', 'fontSize': '11px', 'fontWeight': '600'})
                ], style={'marginBottom': '6px'}),
                html.P([
                    html.Span("r₂: ", style={'color': '#B0BEC5', 'fontSize': '11px'}),
                    html.Span(f"{r2_km:.2f} km", style={'color': '#E0E0E0', 'fontSize': '11px', 'fontWeight': '600'})
                ], style={'marginBottom': '6px'}),
                html.P([
                    html.Span("a_trans: ", style={'color': '#B0BEC5', 'fontSize': '11px'}),
                    html.Span(f"{a_transfer_km:.2f} km", style={'color': '#3498DB', 'fontSize': '12px', 'fontWeight': 'bold'})
                ], style={'marginBottom': '0px'})
            ], style={'padding': '12px', 'backgroundColor': '#263238'})
        ], style={
            'backgroundColor': '#1e1e1e',
            'border': '1px solid #37474F',
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.3)',
            'height': '100%'
        }),
        
        # Card 2: Burn 1 Analysis
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-fire", style={'marginRight': '8px', 'color': '#2ECC71'}),
                html.Span("Burn 1 (Peri)", style={'color': 'white', 'fontWeight': '700', 'fontSize': '13px'})
            ], style={
                'background': 'linear-gradient(135deg, #34495E 0%, #2C3E50 100%)',
                'padding': '10px 12px',
                'borderBottom': '2px solid #2ECC71'
            }),
            dbc.CardBody([
                html.P([
                    html.Span("V₁: ", style={'color': '#B0BEC5', 'fontSize': '11px'}),
                    html.Span(f"{v_initial:.4f} km/s", style={'color': '#E0E0E0', 'fontSize': '11px', 'fontWeight': '600'})
                ], style={'marginBottom': '4px'}),
                html.P([
                    html.Span("V_trans_p: ", style={'color': '#B0BEC5', 'fontSize': '11px'}),
                    html.Span(f"{v_transfer_p:.4f} km/s", style={'color': '#E0E0E0', 'fontSize': '11px', 'fontWeight': '600'})
                ], style={'marginBottom': '6px'}),
                html.Hr(style={'borderColor': '#37474F', 'margin': '6px 0'}),
                html.P(f"ΔV₁ = {abs(delta_v1_val):.4f} km/s", 
                       style={'color': '#2ECC71', 'fontWeight': 'bold', 'fontSize': '13px', 'marginBottom': '0px', 'textAlign': 'center'})
            ], style={'padding': '12px', 'backgroundColor': '#263238'})
        ], style={
            'backgroundColor': '#1e1e1e',
            'border': '1px solid #37474F',
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.3)',
            'height': '100%'
        }),
        
        # Card 3: Burn 2 Analysis
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-fire", style={'marginRight': '8px', 'color': '#2ECC71'}),
                html.Span("Burn 2 (Apo)", style={'color': 'white', 'fontWeight': '700', 'fontSize': '13px'})
            ], style={
                'background': 'linear-gradient(135deg, #34495E 0%, #2C3E50 100%)',
                'padding': '10px 12px',
                'borderBottom': '2px solid #2ECC71'
            }),
            dbc.CardBody([
                html.P([
                    html.Span("V_trans_a: ", style={'color': '#B0BEC5', 'fontSize': '11px'}),
                    html.Span(f"{v_transfer_a:.4f} km/s", style={'color': '#E0E0E0', 'fontSize': '11px', 'fontWeight': '600'})
                ], style={'marginBottom': '4px'}),
                html.P([
                    html.Span("V₂: ", style={'color': '#B0BEC5', 'fontSize': '11px'}),
                    html.Span(f"{v_final:.4f} km/s", style={'color': '#E0E0E0', 'fontSize': '11px', 'fontWeight': '600'})
                ], style={'marginBottom': '6px'}),
                html.Hr(style={'borderColor': '#37474F', 'margin': '6px 0'}),
                html.P(f"ΔV₂ = {abs(delta_v2_val):.4f} km/s", 
                       style={'color': '#2ECC71', 'fontWeight': 'bold', 'fontSize': '13px', 'marginBottom': '0px', 'textAlign': 'center'})
            ], style={'padding': '12px', 'backgroundColor': '#263238'})
        ], style={
            'backgroundColor': '#1e1e1e',
            'border': '1px solid #37474F',
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.3)',
            'height': '100%'
        }),
        
        # Card 4: Total Delta-V Summary
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-tachometer-alt", style={'marginRight': '8px', 'color': '#E74C3C'}),
                html.Span("Total ΔV", style={'color': 'white', 'fontWeight': '700', 'fontSize': '13px'})
            ], style={
                'background': 'linear-gradient(135deg, #E74C3C 0%, #C0392B 100%)',
                'padding': '10px 12px',
                'borderBottom': '2px solid #E74C3C'
            }),
            dbc.CardBody([
                html.P([
                    html.Span("ΔV₁: ", style={'color': '#B0BEC5', 'fontSize': '11px'}),
                    html.Span(f"{abs(delta_v1_val):.4f} km/s", style={'color': '#E0E0E0', 'fontSize': '11px', 'fontWeight': '600'})
                ], style={'marginBottom': '4px'}),
                html.P([
                    html.Span("ΔV₂: ", style={'color': '#B0BEC5', 'fontSize': '11px'}),
                    html.Span(f"{abs(delta_v2_val):.4f} km/s", style={'color': '#E0E0E0', 'fontSize': '11px', 'fontWeight': '600'})
                ], style={'marginBottom': '6px'}),
                html.Hr(style={'borderColor': '#37474F', 'margin': '6px 0'}),
                html.H4(f"{params['total_delta_v'].to(u.km/u.s).value:.4f} km/s", 
                       style={'color': '#E74C3C', 'fontWeight': 'bold', 'fontSize': '18px', 'textAlign': 'center', 'marginBottom': '0px', 'textShadow': '0 2px 4px rgba(231,76,60,0.3)'})
            ], style={'padding': '12px', 'backgroundColor': '#263238'})
        ], style={
            'backgroundColor': '#1e1e1e',
            'border': '2px solid #E74C3C',
            'borderRadius': '10px',
            'boxShadow': '0 4px 15px rgba(231,76,60,0.3)',
            'height': '100%'
        })
    ], style={'height': '100%'})
    
    return fig, mission_table, physics_content


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Starting CubeSat Debris Removal Dashboard...")
    print("="*70)
    print("\nThe web application is starting...")
    print("Once ready, open your browser and go to:")
    print("  http://127.0.0.1:8050")
    print("  or")
    print("  http://localhost:8050")
    print("\nPress Ctrl+C to stop the server.\n")
    print("="*70 + "\n")
    app.run(debug=True, host='127.0.0.1', port=8050)

