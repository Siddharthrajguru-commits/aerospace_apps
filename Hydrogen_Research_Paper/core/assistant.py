"""
Smart Assistant Module for Hydrogen-Electric UAV Propulsion Simulator

This module provides AI-powered analysis and recommendations based on simulation results.
Uses LLM integration to provide intelligent insights and optimization suggestions.

Author: Senior Aerospace Propulsion Engineer
"""

import json
from typing import Dict, Any, Optional


class PropulsionAssistant:
    """
    Smart assistant for propulsion system analysis and recommendations.
    
    Analyzes simulation results and provides intelligent recommendations
    for system optimization based on current performance metrics.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the propulsion assistant.
        
        Args:
            api_key (str, optional): API key for LLM service. If None, uses placeholder mode.
            model (str): Model name for LLM. Default is "gpt-4".
        """
        self.api_key = api_key
        self.model = model
        self.use_llm = api_key is not None
    
    def collect_simulation_state(self, tank_params: Dict[str, Any],
                                 fuel_cell_params: Dict[str, Any],
                                 mission_params: Dict[str, Any],
                                 safety_params: Dict[str, Any],
                                 results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect current simulation state and results.
        
        Args:
            tank_params: Tank configuration parameters
            fuel_cell_params: Fuel cell parameters
            mission_params: Mission profile parameters
            safety_params: Safety compliance parameters
            results: Current simulation results
        
        Returns:
            dict: Complete simulation state dictionary
        """
        state = {
            'tank': tank_params,
            'fuel_cell': fuel_cell_params,
            'mission': mission_params,
            'safety': safety_params,
            'results': results,
            'timestamp': None  # Can add timestamp if needed
        }
        return state
    
    def format_state_for_llm(self, state: Dict[str, Any]) -> str:
        """
        Format simulation state into a prompt for LLM analysis.
        
        Args:
            state: Complete simulation state dictionary
        
        Returns:
            str: Formatted prompt string
        """
        prompt = """You are an expert Aerospace Propulsion Engineer analyzing a hydrogen-electric UAV propulsion system.

CURRENT SYSTEM CONFIGURATION:

Tank Parameters:
- Radius: {tank_radius:.3f} m
- Length: {tank_length:.3f} m
- Wall Thickness: {wall_thickness_mm:.2f} mm
- Insulation Thickness: {insulation_thickness_mm:.1f} mm
- Operating Pressure: {operating_pressure_kpa:.1f} kPa
- Material Yield Strength: {material_strength_mpa:.0f} MPa

Fuel Cell Parameters:
- Stack Power: {fuel_cell_power_kw:.1f} kW
- Open Circuit Voltage: {e_ocv:.2f} V
- Operating Temperature: {temperature:.1f} K

Mission Parameters:
- Payload Mass: {payload_mass:.1f} kg
- Fuel Mass: {fuel_mass:.2f} kg
- Cruise Velocity: {cruise_velocity:.1f} m/s
- L/D Ratio: {lift_to_drag:.1f}

CURRENT RESULTS:
- Mission Range: {range_km:.2f} km
- Heat Leak: {heat_leak:.2f} W
- Boil-Off Rate: {boil_off_rate:.4f} kg/h
- Factor of Safety: {fos:.2f} (EASA Minimum: 2.2)
- Reserve Fuel Required: {reserve_fuel_g:.1f} g
- Usable Range: {usable_range_km:.2f} km

ANALYSIS REQUEST:
{user_query}

Please provide:
1. Analysis of current system performance
2. Identification of any issues or concerns
3. Specific, actionable recommendations with quantitative suggestions
4. Priority ranking of recommendations

Format your response as a clear, professional engineering analysis suitable for a Master's research portfolio.
""".format(
            tank_radius=state['tank'].get('radius', 0.3),
            tank_length=state['tank'].get('length', 1.5),
            wall_thickness_mm=state['tank'].get('wall_thickness_mm', 2.0),
            insulation_thickness_mm=state['tank'].get('insulation_thickness_mm', 50.0),
            operating_pressure_kpa=state['tank'].get('operating_pressure_kpa', 500.0),
            material_strength_mpa=state['tank'].get('material_strength_mpa', 350.0),
            fuel_cell_power_kw=state['fuel_cell'].get('power_kw', 5.0),
            e_ocv=state['fuel_cell'].get('e_ocv', 1.2),
            temperature=state['fuel_cell'].get('temperature', 353.15),
            payload_mass=state['mission'].get('payload_mass', 5.0),
            fuel_mass=state['mission'].get('fuel_mass', 2.0),
            cruise_velocity=state['mission'].get('cruise_velocity', 30.0),
            lift_to_drag=state['mission'].get('lift_to_drag', 15.0),
            range_km=state['results'].get('range_km', 150.0),
            heat_leak=state['results'].get('heat_leak', 10.0),
            boil_off_rate=state['results'].get('boil_off_rate', 0.001),
            fos=state['results'].get('fos', 2.5),
            reserve_fuel_g=state['results'].get('reserve_fuel_g', 50.0),
            usable_range_km=state['results'].get('usable_range_km', 140.0),
            user_query=state.get('user_query', 'Analyze the current system configuration and provide recommendations.')
        )
        
        return prompt
    
    def call_llm_api(self, prompt: str) -> str:
        """
        Call LLM API for analysis (placeholder implementation).
        
        This is a placeholder that can be replaced with actual API calls to:
        - OpenAI GPT-4
        - Anthropic Claude
        - Local LLM via API
        - Other LLM services
        
        Args:
            prompt: Formatted prompt string
        
        Returns:
            str: LLM response text
        """
        if not self.use_llm:
            # Placeholder: Rule-based analysis
            return self._rule_based_analysis(prompt)
        
        # TODO: Implement actual LLM API call
        # Example structure:
        # import openai
        # response = openai.ChatCompletion.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.7
        # )
        # return response.choices[0].message.content
        
        return self._rule_based_analysis(prompt)
    
    def _rule_based_analysis(self, prompt: str) -> str:
        """
        Rule-based analysis as fallback when LLM API is not available.
        
        Provides intelligent recommendations based on simulation parameters.
        
        Args:
            prompt: Formatted prompt (contains system state)
        
        Returns:
            str: Analysis and recommendations
        """
        # Extract key parameters from prompt (simplified parsing)
        # In production, would parse the structured state dict
        
        # Parse numeric values from prompt
        import re
        
        # Extract key metrics
        boil_off_rate_match = re.search(r'Boil-Off Rate: ([\d.]+)', prompt)
        insulation_match = re.search(r'Insulation Thickness: ([\d.]+)', prompt)
        fos_match = re.search(r'Factor of Safety: ([\d.]+)', prompt)
        range_match = re.search(r'Mission Range: ([\d.]+)', prompt)
        heat_leak_match = re.search(r'Heat Leak: ([\d.]+)', prompt)
        
        boil_off_rate = float(boil_off_rate_match.group(1)) if boil_off_rate_match else 0.001
        insulation_thickness = float(insulation_match.group(1)) if insulation_match else 50.0
        fos = float(fos_match.group(1)) if fos_match else 2.5
        mission_range = float(range_match.group(1)) if range_match else 150.0
        heat_leak = float(heat_leak_match.group(1)) if heat_leak_match else 10.0
        
        recommendations = []
        analysis = []
        
        # Analyze boil-off rate
        if boil_off_rate > 0.002:  # High boil-off (> 2 g/h)
            analysis.append(f"⚠️ **High Boil-Off Rate Detected**: {boil_off_rate*1000:.2f} g/h")
            if insulation_thickness < 60:
                increase = min(5, (0.002 - boil_off_rate) * 1000 * 2)  # Estimate
                recommendations.append(
                    f"**Recommendation 1**: Increase MLI thickness by {increase:.1f} mm "
                    f"(from {insulation_thickness:.1f} mm to {insulation_thickness + increase:.1f} mm) "
                    f"to reduce boil-off losses. This will reduce heat leak and improve fuel retention."
                )
            else:
                recommendations.append(
                    "**Recommendation 1**: Consider advanced insulation materials or vacuum insulation "
                    "to further reduce boil-off rate."
                )
        elif boil_off_rate < 0.0005:  # Very low boil-off
            analysis.append(f"✅ **Excellent Boil-Off Rate**: {boil_off_rate*1000:.2f} g/h")
        
        # Analyze Factor of Safety
        if fos < 2.2:
            analysis.append(f"❌ **CRITICAL: Factor of Safety Below EASA Minimum**: {fos:.2f} < 2.2")
            recommendations.append(
                "**Recommendation 2 (HIGH PRIORITY)**: Increase tank wall thickness or use higher "
                "strength material to meet EASA Factor of Safety requirement of 2.2. "
                "Current design is NON-COMPLIANT and unsafe for flight."
            )
        elif fos < 2.5:
            analysis.append(f"⚠️ **Low Safety Margin**: FoS = {fos:.2f}")
            recommendations.append(
                "**Recommendation 2**: Consider increasing wall thickness by 0.2-0.5 mm to improve "
                "safety margin above EASA minimum."
            )
        else:
            analysis.append(f"✅ **Adequate Safety Margin**: FoS = {fos:.2f}")
        
        # Analyze heat leak
        if heat_leak > 15:
            analysis.append(f"⚠️ **High Heat Leak**: {heat_leak:.2f} W")
            recommendations.append(
                "**Recommendation 3**: Optimize MLI configuration or increase insulation thickness "
                "to reduce thermal losses."
            )
        
        # Analyze mission range
        if mission_range < 100:
            analysis.append(f"⚠️ **Limited Mission Range**: {mission_range:.2f} km")
            recommendations.append(
                "**Recommendation 4**: Consider increasing fuel mass or optimizing system efficiency "
                "to extend mission range. Current range may limit operational flexibility."
            )
        
        # Compile response
        response = f"""## System Analysis Report

### Current Performance Summary

{chr(10).join(analysis)}

### Detailed Recommendations

{chr(10).join(recommendations) if recommendations else "✅ **System Performance**: All parameters within acceptable ranges. No critical issues identified."}

### Optimization Priority

1. **Safety Compliance** (if FoS < 2.2): Address immediately - non-compliance prevents certification
2. **Boil-Off Rate**: Critical for long-duration missions (>2 hours)
3. **Heat Leak**: Affects fuel retention and system efficiency
4. **Mission Range**: Operational capability consideration

### Next Steps

- Review recommendations above and adjust system parameters accordingly
- Re-run simulation to verify improvements
- Consider trade-offs between weight, performance, and safety

---
*Analysis generated by Propulsion System Assistant*
"""
        
        return response
    
    def analyze_and_recommend(self, user_query: str, simulation_state: Dict[str, Any]) -> str:
        """
        Main method to analyze simulation state and provide recommendations.
        
        Args:
            user_query: User's question or request
            simulation_state: Complete simulation state dictionary
        
        Returns:
            str: Analysis and recommendations text
        """
        # Add user query to state
        simulation_state['user_query'] = user_query
        
        # Format state for LLM
        prompt = self.format_state_for_llm(simulation_state)
        
        # Call LLM (or rule-based fallback)
        response = self.call_llm_api(prompt)
        
        return response
