"""
Industry Benchmark Specifications for Hydrogen-Electric Aircraft

This module contains technical specifications for industry-leading hydrogen-electric
aircraft designs, including the AeroDelft Phoenix (TU Delft) and Airbus ZEROe regional concept.

These benchmarks are used for comparative analysis to evaluate UAV design performance
against established industry standards.

Author: Senior Aerospace Propulsion Engineer
"""


class IndustryBenchmark:
    """
    Base class for industry benchmark specifications.
    
    Contains standardized metrics for comparing hydrogen-electric propulsion systems:
    - System efficiency (total propulsion efficiency)
    - Energy density (effective energy density including tank/system mass)
    - Scale factors (mass, power, range)
    """
    
    def __init__(self, name, system_efficiency, energy_density_wh_per_kg, 
                 total_mass_kg, fuel_mass_kg, payload_mass_kg, 
                 max_range_km, max_power_kw, lift_to_drag, 
                 description="", source=""):
        """
        Initialize benchmark specifications.
        
        Args:
            name (str): Aircraft name/model
            system_efficiency (float): Total system efficiency (0-1)
            energy_density_wh_per_kg (float): Effective energy density in Wh/kg
                                              (including tank/system mass)
            total_mass_kg (float): Maximum takeoff mass in kg
            fuel_mass_kg (float): Maximum fuel mass in kg
            payload_mass_kg (float): Payload capacity in kg
            max_range_km (float): Maximum range in km
            max_power_kw (float): Maximum power output in kW
            lift_to_drag (float): Lift-to-drag ratio
            description (str): Description of the aircraft
            source (str): Source/reference for specifications
        """
        self.name = name
        self.system_efficiency = system_efficiency
        self.energy_density_wh_per_kg = energy_density_wh_per_kg
        self.total_mass_kg = total_mass_kg
        self.fuel_mass_kg = fuel_mass_kg
        self.payload_mass_kg = payload_mass_kg
        self.max_range_km = max_range_km
        self.max_power_kw = max_power_kw
        self.lift_to_drag = lift_to_drag
        self.description = description
        self.source = source
        
        # Calculate effective energy density (J/kg) including system mass
        self.energy_density_j_per_kg = energy_density_wh_per_kg * 3600  # Convert Wh/kg to J/kg
        
        # Calculate specific energy (energy per unit system mass)
        self.specific_energy_wh_per_kg_system = (
            fuel_mass_kg * energy_density_wh_per_kg / total_mass_kg
        )
    
    def get_metrics_dict(self):
        """Return benchmark metrics as dictionary."""
        return {
            'name': self.name,
            'system_efficiency': self.system_efficiency,
            'energy_density_wh_per_kg': self.energy_density_wh_per_kg,
            'energy_density_j_per_kg': self.energy_density_j_per_kg,
            'total_mass_kg': self.total_mass_kg,
            'fuel_mass_kg': self.fuel_mass_kg,
            'payload_mass_kg': self.payload_mass_kg,
            'max_range_km': self.max_range_km,
            'max_power_kw': self.max_power_kw,
            'lift_to_drag': self.lift_to_drag,
            'specific_energy_wh_per_kg_system': self.specific_energy_wh_per_kg_system,
            'description': self.description,
            'source': self.source
        }


# AeroDelft Phoenix (TU Delft) Specifications
# Reference: AeroDelft Phoenix Project - Student-led hydrogen-electric aircraft
# https://www.aerodelft.nl/phoenix
AERODELFT_PHOENIX = IndustryBenchmark(
    name="AeroDelft Phoenix",
    system_efficiency=0.52,  # 52% total system efficiency (fuel cell + motor + propulsive)
    energy_density_wh_per_kg=850,  # Effective energy density including tank mass
    total_mass_kg=850.0,  # Maximum takeoff mass
    fuel_mass_kg=25.0,  # Liquid hydrogen fuel mass
    payload_mass_kg=150.0,  # Payload capacity (pilot + equipment)
    max_range_km=2000.0,  # Maximum range
    max_power_kw=80.0,  # Maximum power output
    lift_to_drag=18.0,  # High L/D ratio for efficient glider design
    description=(
        "AeroDelft Phoenix is a student-led project at TU Delft developing a "
        "hydrogen-electric aircraft. It features advanced fuel cell technology "
        "and optimized aerodynamics for long-range flight. The Phoenix demonstrates "
        "the potential of hydrogen-electric propulsion for sustainable aviation."
    ),
    source="AeroDelft Phoenix Project, TU Delft (https://www.aerodelft.nl/phoenix)"
)

# Airbus ZEROe Regional Concept Specifications
# Reference: Airbus ZEROe concept aircraft - Regional variant
# https://www.airbus.com/en/innovation/zero-emission/zeroe
AIRBUS_ZEROE_REGIONAL = IndustryBenchmark(
    name="Airbus ZEROe Regional",
    system_efficiency=0.55,  # 55% total system efficiency (advanced fuel cell + motor)
    energy_density_wh_per_kg=720,  # Effective energy density (larger scale, more conservative)
    total_mass_kg=45000.0,  # Maximum takeoff mass (regional aircraft scale)
    fuel_mass_kg=2000.0,  # Liquid hydrogen fuel mass
    payload_mass_kg=9000.0,  # Payload capacity (passengers + cargo)
    max_range_km=1800.0,  # Maximum range
    max_power_kw=2000.0,  # Maximum power output
    lift_to_drag=16.5,  # L/D ratio for regional aircraft
    description=(
        "Airbus ZEROe Regional is a conceptual hydrogen-electric regional aircraft "
        "designed for short-to-medium range flights. It represents Airbus's vision "
        "for zero-emission aviation using liquid hydrogen fuel cells. The regional "
        "variant targets 50-100 passenger capacity with ranges up to 1800 km."
    ),
    source="Airbus ZEROe Concept Aircraft (https://www.airbus.com/en/innovation/zero-emission/zeroe)"
)


def get_all_benchmarks():
    """
    Get all available industry benchmarks.
    
    Returns:
        list: List of IndustryBenchmark objects
    """
    return [AERODELFT_PHOENIX, AIRBUS_ZEROE_REGIONAL]


def get_benchmark_by_name(name):
    """
    Get a specific benchmark by name.
    
    Args:
        name (str): Benchmark name
        
    Returns:
        IndustryBenchmark: Benchmark object or None if not found
    """
    benchmarks = get_all_benchmarks()
    for benchmark in benchmarks:
        if benchmark.name.lower() == name.lower():
            return benchmark
    return None


def calculate_delta_analysis(uav_efficiency, uav_energy_density_wh_per_kg, 
                              uav_mass_kg, benchmark):
    """
    Calculate delta (difference) analysis between UAV and benchmark.
    
    Args:
        uav_efficiency (float): UAV system efficiency (0-1)
        uav_energy_density_wh_per_kg (float): UAV effective energy density in Wh/kg
        uav_mass_kg (float): UAV total mass in kg
        benchmark (IndustryBenchmark): Benchmark to compare against
        
    Returns:
        dict: Dictionary containing delta metrics and analysis
    """
    # Calculate deltas
    delta_efficiency = uav_efficiency - benchmark.system_efficiency
    delta_efficiency_pct = (delta_efficiency / benchmark.system_efficiency) * 100
    
    delta_energy_density = uav_energy_density_wh_per_kg - benchmark.energy_density_wh_per_kg
    delta_energy_density_pct = (delta_energy_density / benchmark.energy_density_wh_per_kg) * 100
    
    # Scale factor (mass ratio)
    scale_factor = uav_mass_kg / benchmark.total_mass_kg
    
    # Generate analysis text
    analysis_parts = []
    
    # Efficiency analysis
    if abs(delta_efficiency_pct) < 5:
        efficiency_comment = (
            f"Your UAV's efficiency ({uav_efficiency*100:.1f}%) is very close to "
            f"{benchmark.name} ({benchmark.system_efficiency*100:.1f}%), indicating "
            f"similar fuel cell and motor technology maturity."
        )
    elif delta_efficiency_pct > 0:
        efficiency_comment = (
            f"Your UAV shows {abs(delta_efficiency_pct):.1f}% higher efficiency than "
            f"{benchmark.name}. This may be due to optimized fuel cell operating conditions "
            f"or advanced motor technology."
        )
    else:
        efficiency_comment = (
            f"Your UAV shows {abs(delta_efficiency_pct):.1f}% lower efficiency than "
            f"{benchmark.name}. This difference could be due to scale effects (smaller "
            f"systems often have lower efficiency), different fuel cell technology, "
            f"or operating conditions."
        )
    
    # Energy density analysis
    if abs(delta_energy_density_pct) < 10:
        energy_comment = (
            f"Energy density is comparable ({uav_energy_density_wh_per_kg:.0f} vs "
            f"{benchmark.energy_density_wh_per_kg:.0f} Wh/kg), suggesting similar "
            f"tank technology and system integration."
        )
    elif delta_energy_density_pct > 0:
        energy_comment = (
            f"Your UAV achieves {abs(delta_energy_density_pct):.1f}% higher energy density "
            f"({uav_energy_density_wh_per_kg:.0f} vs {benchmark.energy_density_wh_per_kg:.0f} Wh/kg). "
            f"This advantage may come from lighter tank design or optimized system mass."
        )
    else:
        energy_comment = (
            f"Your UAV shows {abs(delta_energy_density_pct):.1f}% lower energy density "
            f"({uav_energy_density_wh_per_kg:.0f} vs {benchmark.energy_density_wh_per_kg:.0f} Wh/kg). "
            f"This is expected due to scale effects - smaller systems typically have "
            f"higher tank-to-fuel mass ratios, reducing effective energy density."
        )
    
    # Scale effects analysis
    if scale_factor < 0.1:
        scale_comment = (
            f"**Scale Effects**: Your UAV is {scale_factor:.2%} the mass of {benchmark.name} "
            f"({uav_mass_kg:.1f} vs {benchmark.total_mass_kg:.0f} kg). At this scale, "
            f"you face challenges including:\n"
            f"- Higher tank-to-fuel mass ratios (tank structure overhead)\n"
            f"- Lower fuel cell efficiency (smaller stacks)\n"
            f"- Reduced aerodynamic efficiency (Reynolds number effects)\n"
            f"- Higher specific power requirements\n\n"
            f"Despite these challenges, small-scale UAVs benefit from lower absolute "
            f"power requirements and can achieve competitive specific performance metrics."
        )
    elif scale_factor < 0.5:
        scale_comment = (
            f"**Scale Effects**: Your UAV is {scale_factor:.1%} the mass of {benchmark.name} "
            f"({uav_mass_kg:.1f} vs {benchmark.total_mass_kg:.0f} kg). At this intermediate "
            f"scale, you benefit from some economies of scale while maintaining flexibility "
            f"for optimization."
        )
    else:
        scale_comment = (
            f"**Scale Effects**: Your UAV is {scale_factor:.1%} the mass of {benchmark.name} "
            f"({uav_mass_kg:.1f} vs {benchmark.total_mass_kg:.0f} kg). At similar scales, "
            f"performance differences are primarily due to design choices rather than "
            f"fundamental scale limitations."
        )
    
    # Power assumptions
    power_comment = (
        f"**Power Assumptions**: {benchmark.name} operates at {benchmark.max_power_kw:.0f} kW "
        f"maximum power. Your UAV's power requirements scale with mass and cruise speed. "
        f"Smaller aircraft typically require lower absolute power but may have higher "
        f"specific power (W/kg) due to less favorable scaling of aerodynamic drag."
    )
    
    analysis_parts.extend([
        efficiency_comment,
        energy_comment,
        scale_comment,
        power_comment
    ])
    
    return {
        'delta_efficiency': delta_efficiency,
        'delta_efficiency_pct': delta_efficiency_pct,
        'delta_energy_density': delta_energy_density,
        'delta_energy_density_pct': delta_energy_density_pct,
        'scale_factor': scale_factor,
        'analysis_text': '\n\n'.join(analysis_parts),
        'benchmark_name': benchmark.name
    }
