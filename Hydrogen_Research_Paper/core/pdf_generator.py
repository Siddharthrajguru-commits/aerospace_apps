"""
PDF Generation Module for Technical Memorandums

Generates NASA-style technical memorandums in PDF format for propulsion system analysis.

Author: Senior Aerospace Propulsion Engineer
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import io

# Try to import reportlab, handle gracefully if not installed
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    # Create dummy classes to prevent errors
    letter = None
    inch = None
    colors = None
    getSampleStyleSheet = None
    ParagraphStyle = None
    SimpleDocTemplate = None
    Paragraph = None
    Spacer = None
    Table = None
    TableStyle = None
    PageBreak = None
    TA_CENTER = None
    TA_LEFT = None
    TA_JUSTIFY = None


def generate_nasa_technical_memo(simulation_state: Dict[str, Any], 
                                  assistant_response: str,
                                  output_path: str = None) -> Optional[bytes]:
    """
    Generate a NASA-style technical memorandum PDF.
    
    Args:
        simulation_state: Complete simulation state dictionary
        assistant_response: AI assistant analysis and recommendations
        output_path: Optional file path to save PDF. If None, returns bytes.
        
    Returns:
        bytes: PDF file as bytes if output_path is None, or None if reportlab not available
        
    Raises:
        ImportError: If reportlab is not installed
    """
    # Check if reportlab is available
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is not installed. Please install it using: pip install reportlab\n"
            "The PDF generation feature requires reportlab>=3.6.0"
        )
    
    # Create PDF buffer
    buffer = io.BytesIO()
    
    # Create document
    doc = SimpleDocTemplate(
        output_path if output_path else buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Container for PDF elements
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles for NASA memo
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=6,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=10,
        textColor=colors.black,
        spaceAfter=4,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    # NASA Header
    story.append(Paragraph("NATIONAL AERONAUTICS AND SPACE ADMINISTRATION", 
                          ParagraphStyle('NASAHeader', parent=styles['Normal'], 
                                        fontSize=9, alignment=TA_CENTER, 
                                        fontName='Helvetica-Bold')))
    story.append(Spacer(1, 0.1*inch))
    
    # Title
    story.append(Paragraph(
        "Hydrogen-Electric UAV Propulsion System Analysis",
        title_style
    ))
    story.append(Spacer(1, 0.1*inch))
    
    # Document info
    doc_info = f"""
    <b>Technical Memorandum</b><br/>
    Date: {datetime.now().strftime('%B %d, %Y')}<br/>
    Prepared by: Propulsion System Analysis Tool<br/>
    """
    story.append(Paragraph(doc_info, ParagraphStyle('DocInfo', parent=styles['Normal'], 
                                                   fontSize=9, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.2*inch))
    
    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    
    exec_summary = f"""
    This technical memorandum presents a comprehensive analysis of a hydrogen-electric 
    unmanned aerial vehicle (UAV) propulsion system. The analysis evaluates system 
    performance, safety compliance, and optimization opportunities based on current 
    design parameters.
    
    The propulsion system utilizes liquid hydrogen (LH₂) fuel stored in a cryogenic tank 
    with multi-layer insulation (MLI), powering a proton exchange membrane (PEM) fuel 
    cell stack. Key performance metrics include mission range, thermal management 
    efficiency, and safety compliance with European Union Aviation Safety Agency (EASA) 
    standards.
    
    Current system configuration achieves a mission range of {simulation_state['results'].get('range_km', 0):.2f} km 
    with a boil-off rate of {simulation_state['results'].get('boil_off_rate_g_per_h', 0)*1000:.4f} g/h. 
    The system demonstrates {'compliance' if simulation_state['safety'].get('is_compliant', False) else 'non-compliance'} 
    with EASA Factor of Safety requirements (FoS = {simulation_state['safety'].get('fos', 0):.2f}).
    """
    
    story.append(Paragraph(exec_summary, normal_style))
    story.append(Spacer(1, 0.15*inch))
    
    # Optimized Parameters Table
    story.append(Paragraph("SYSTEM PARAMETERS", heading_style))
    
    # Extract parameters
    tank_params = simulation_state.get('tank', {})
    fuel_cell_params = simulation_state.get('fuel_cell', {})
    mission_params = simulation_state.get('mission', {})
    results = simulation_state.get('results', {})
    
    # Create parameter table
    param_data = [
        ['Parameter', 'Value', 'Units'],
        ['<b>Tank Configuration</b>', '', ''],
        ['Tank Radius', f"{tank_params.get('radius', 0):.3f}", 'm'],
        ['Tank Length', f"{tank_params.get('length', 0):.3f}", 'm'],
        ['Wall Thickness', f"{tank_params.get('wall_thickness_mm', 0):.2f}", 'mm'],
        ['Insulation Thickness', f"{tank_params.get('insulation_thickness_mm', 0):.1f}", 'mm'],
        ['Operating Pressure', f"{tank_params.get('operating_pressure_kpa', 0):.1f}", 'kPa'],
        ['Material Yield Strength', f"{tank_params.get('material_strength_mpa', 0):.0f}", 'MPa'],
        ['<b>Fuel Cell System</b>', '', ''],
        ['Stack Power', f"{fuel_cell_params.get('power_kw', 0):.1f}", 'kW'],
        ['Open Circuit Voltage', f"{fuel_cell_params.get('e_ocv', 0):.2f}", 'V'],
        ['Operating Temperature', f"{fuel_cell_params.get('temperature', 0):.1f}", 'K'],
        ['<b>Mission Profile</b>', '', ''],
        ['Payload Mass', f"{mission_params.get('payload_mass', 0):.1f}", 'kg'],
        ['Fuel Mass', f"{mission_params.get('fuel_mass', 0):.2f}", 'kg'],
        ['Cruise Velocity', f"{mission_params.get('cruise_velocity', 0):.1f}", 'm/s'],
        ['Lift-to-Drag Ratio', f"{mission_params.get('lift_to_drag', 0):.1f}", ''],
        ['Total System Efficiency', f"{mission_params.get('total_efficiency', 0)*100:.1f}", '%'],
        ['<b>Performance Results</b>', '', ''],
        ['Mission Range', f"{results.get('range_km', 0):.2f}", 'km'],
        ['Heat Leak Rate', f"{results.get('heat_leak', 0):.2f}", 'W'],
        ['Boil-Off Rate', f"{results.get('boil_off_rate_g_per_h', results.get('boil_off_rate', 0)*3600*1000):.4f}", 'g/h'],
        ['Factor of Safety', f"{simulation_state['safety'].get('fos', 0):.2f}", ''],
        ['Reserve Fuel Required', f"{results.get('reserve_fuel_g', 0):.1f}", 'g'],
        ['Usable Range', f"{results.get('usable_range_km', 0):.2f}", 'km'],
    ]
    
    param_table = Table(param_data, colWidths=[3*inch, 1.5*inch, 1*inch])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    story.append(param_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Page break before risks section
    story.append(PageBreak())
    
    # Risks and Mitigations
    story.append(Paragraph("RISKS AND MITIGATIONS", heading_style))
    
    # Extract risks from assistant response and system state
    risks = []
    
    # Factor of Safety risk
    safety_data = simulation_state.get('safety', {})
    fos = safety_data.get('fos', 0)
    is_compliant = safety_data.get('is_compliant', True)
    
    if fos < 2.2 or not is_compliant:
        risks.append({
            'risk': 'Non-Compliance with EASA Factor of Safety Requirements',
            'severity': 'CRITICAL',
            'description': f'The current Factor of Safety ({fos:.2f}) is below the EASA minimum requirement of 2.2. This represents a critical safety risk that prevents system certification and flight operations.',
            'mitigation': [
                'Increase tank wall thickness by 0.2-0.5 mm to improve structural safety margin',
                'Consider using higher strength materials (e.g., titanium alloy with yield strength >450 MPa)',
                'Reduce operating pressure if mission requirements allow',
                'Conduct finite element analysis (FEA) to optimize wall thickness distribution'
            ]
        })
    elif fos < 2.5:
        risks.append({
            'risk': 'Low Safety Margin Above EASA Minimum',
            'severity': 'HIGH',
            'description': f'Factor of Safety ({fos:.2f}) is above minimum but provides limited margin for manufacturing variations and operational uncertainties.',
            'mitigation': [
                'Increase wall thickness by 0.1-0.3 mm to provide additional safety margin',
                'Implement strict quality control procedures for tank manufacturing',
                'Consider design margin for fatigue and environmental factors'
            ]
        })
    
    # Boil-off risk
    # Handle both formats: boil_off_rate_g_per_h (already in g/h) or boil_off_rate (in kg/s)
    if 'boil_off_rate_g_per_h' in results:
        boil_off_rate = results.get('boil_off_rate_g_per_h', 0)  # Already in g/h
    else:
        # Convert from kg/s to g/h
        boil_off_rate = results.get('boil_off_rate', 0) * 3600 * 1000
    
    if boil_off_rate > 2.0:
        risks.append({
            'risk': 'Excessive Boil-Off Rate',
            'severity': 'HIGH',
            'description': f'Boil-off rate ({boil_off_rate:.4f} g/h) exceeds acceptable limits for long-duration missions. This results in significant fuel loss and reduced mission range.',
            'mitigation': [
                f'Increase MLI insulation thickness from {tank_params.get("insulation_thickness_mm", 0):.1f} mm to {tank_params.get("insulation_thickness_mm", 0) + 10:.1f} mm',
                'Implement advanced insulation techniques (e.g., vacuum insulation panels)',
                'Optimize tank geometry to minimize surface area-to-volume ratio',
                'Consider active cooling systems for extended mission durations'
            ]
        })
    elif boil_off_rate > 1.0:
        risks.append({
            'risk': 'Moderate Boil-Off Rate',
            'severity': 'MEDIUM',
            'description': f'Boil-off rate ({boil_off_rate:.4f} g/h) may impact fuel availability for extended missions.',
            'mitigation': [
                'Monitor boil-off rate during mission planning',
                'Consider incremental insulation improvements',
                'Plan fuel reserves accounting for boil-off losses'
            ]
        })
    
    # Heat leak risk
    heat_leak = results.get('heat_leak', 0)
    if heat_leak > 15:
        risks.append({
            'risk': 'High Thermal Heat Leak',
            'severity': 'MEDIUM',
            'description': f'Heat leak rate ({heat_leak:.2f} W) contributes to increased boil-off and reduced system efficiency.',
            'mitigation': [
                'Optimize MLI layer configuration and installation',
                'Ensure proper vacuum maintenance in insulation space',
                'Consider radiation shields to reduce radiative heat transfer',
                'Evaluate thermal isolation of tank mounting points'
            ]
        })
    
    # Mission range risk
    mission_range = results.get('range_km', 0)
    if mission_range < 100:
        risks.append({
            'risk': 'Limited Mission Range',
            'severity': 'MEDIUM',
            'description': f'Mission range ({mission_range:.2f} km) may limit operational flexibility and mission capabilities.',
            'mitigation': [
                'Increase fuel mass capacity if payload allows',
                'Optimize system efficiency (fuel cell, motor, propulsive)',
                'Improve aerodynamic efficiency (L/D ratio)',
                'Consider hybrid power systems for extended range'
            ]
        })
    
    # Cryogenic storage safety risks (always include)
    risks.append({
        'risk': 'Cryogenic Storage Safety Hazards',
        'severity': 'HIGH',
        'description': 'Liquid hydrogen storage at 20 K presents multiple safety hazards including embrittlement, pressure buildup, and hydrogen flammability.',
        'mitigation': [
            'Implement comprehensive leak detection systems with hydrogen sensors',
            'Design pressure relief valves with appropriate set points',
            'Use materials compatible with cryogenic temperatures (avoid embrittlement)',
            'Establish safe operating procedures for fueling and maintenance',
            'Implement emergency venting systems for overpressure scenarios',
            'Conduct regular inspections and maintenance of cryogenic systems',
            'Train personnel on hydrogen safety protocols (NFPA 2 standards)',
            'Ensure proper ventilation in storage and operational areas'
        ]
    })
    
    # Add risks to document
    for i, risk in enumerate(risks, 1):
        story.append(Paragraph(
            f"<b>{i}. {risk['risk']}</b>",
            subheading_style
        ))
        
        severity_color = {
            'CRITICAL': colors.HexColor('#d32f2f'),
            'HIGH': colors.HexColor('#f57c00'),
            'MEDIUM': colors.HexColor('#fbc02d'),
            'LOW': colors.HexColor('#388e3c')
        }.get(risk['severity'], colors.black)
        
        story.append(Paragraph(
            f"<b>Severity:</b> <font color='{severity_color.hexColor()}'>{risk['severity']}</font>",
            ParagraphStyle('Severity', parent=styles['Normal'], fontSize=9, fontName='Helvetica-Bold')
        ))
        
        story.append(Paragraph(
            f"<b>Description:</b> {risk['description']}",
            normal_style
        ))
        
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("<b>Mitigation Strategies:</b>", 
                               ParagraphStyle('MitigationHeader', parent=styles['Normal'], 
                                             fontSize=9, fontName='Helvetica-Bold')))
        
        for mitigation in risk['mitigation']:
            story.append(Paragraph(
                f"• {mitigation}",
                ParagraphStyle('Mitigation', parent=styles['Normal'], 
                             fontSize=9, leftIndent=0.2*inch, spaceAfter=3)
            ))
        
        story.append(Spacer(1, 0.15*inch))
    
    # Recommendations Summary (from assistant response)
    story.append(Paragraph("RECOMMENDATIONS SUMMARY", heading_style))
    
    # Extract key recommendations from assistant response
    rec_text = """
    Based on the comprehensive system analysis, the following recommendations are prioritized:
    
    <b>Priority 1 - Safety Compliance:</b> Address Factor of Safety requirements immediately 
    if below EASA minimum (2.2). Non-compliance prevents system certification.
    
    <b>Priority 2 - Thermal Management:</b> Optimize insulation thickness to balance boil-off 
    rate and tank weight. Target boil-off rate <1.0 g/h for extended missions.
    
    <b>Priority 3 - System Optimization:</b> Evaluate trade-offs between fuel mass, 
    insulation weight, and mission range to identify optimal configuration.
    
    <b>Priority 4 - Operational Planning:</b> Account for boil-off losses and reserve fuel 
    requirements in mission planning to ensure safe operations.
    """
    
    story.append(Paragraph(rec_text, normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Footer
    footer_text = f"""
    <i>This technical memorandum was generated automatically by the Hydrogen-Electric UAV 
    Propulsion System Analysis Tool on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}.</i>
    """
    story.append(Paragraph(footer_text, 
                           ParagraphStyle('Footer', parent=styles['Normal'], 
                                         fontSize=8, alignment=TA_CENTER, 
                                         textColor=colors.grey)))
    
    # Build PDF
    doc.build(story)
    
    # Return bytes if no output path
    if output_path is None:
        buffer.seek(0)
        return buffer.getvalue()
    
    return None
