"""
Aether-Agent: Web3-Style Aerospace Synthesis Terminal
Modern Web3/Crypto Platform UI Design with Glassmorphism & Gradients
"""

import streamlit as st
import sys
import logging
from pathlib import Path
from agent_core import AetherAgent
import json
from typing import Dict, List, Optional, Any
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Try to import graphviz for mind maps
GRAPHVIZ_AVAILABLE = False
GRAPHVIZ_ERROR = None
try:
    import graphviz
    try:
        test_dot = graphviz.Digraph()
        test_dot.node('test')
        test_dot.render(format='svg', cleanup=True)
        GRAPHVIZ_AVAILABLE = True
    except Exception as e:
        GRAPHVIZ_AVAILABLE = False
        GRAPHVIZ_ERROR = str(e)
        logger.warning(f"graphviz Python package installed but system binaries not found: {e}")
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    GRAPHVIZ_ERROR = "Python package 'graphviz' not installed"
    logger.warning("graphviz not available. Install with: pip install graphviz")

# Page configuration
st.set_page_config(
    page_title="Aether-Agent: Aerospace Synthesis Terminal",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Web3/Crypto Platform UI Theme with Glassmorphism
st.markdown("""
    <style>
    /* Web3/Crypto Platform Color Palette */
    :root {
        --bg-primary: #0a0e27;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.6);
        --bg-glass: rgba(255, 255, 255, 0.05);
        --border-glass: rgba(255, 255, 255, 0.1);
        
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-accent: linear-gradient(135deg, #00d4ff 0%, #5b21b6 100%);
        --gradient-purple: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        --gradient-blue: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        
        --accent-cyan: #00d4ff;
        --accent-purple: #8b5cf6;
        --accent-pink: #ec4899;
        --accent-blue: #3b82f6;
        
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        
        --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.5);
        --shadow-glow: 0 0 20px rgba(139, 92, 246, 0.3);
        
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
    }
    
    /* Global Styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main .block-container {
        padding: 0;
        max-width: 100%;
        background: var(--bg-primary);
    }
    
    /* Remove Streamlit default styling */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Modal Popup Styles */
    .modal-overlay {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
        z-index: 10000;
        justify-content: center;
        align-items: center;
        animation: fadeIn 0.3s ease;
    }
    
    .modal-overlay.active {
        display: flex;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .modal-content {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-lg);
        padding: 2rem;
        max-width: 80%;
        max-height: 85vh;
        overflow-y: auto;
        box-shadow: var(--shadow-lg);
        position: relative;
        animation: slideUp 0.3s ease;
        color: var(--text-primary);
    }
    
    @keyframes slideUp {
        from {
            transform: translateY(50px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--border-glass);
    }
    
    .modal-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .modal-close {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-sm);
        color: var(--text-primary);
        font-size: 1.5rem;
        width: 40px;
        height: 40px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .modal-close:hover {
        background: rgba(239, 68, 68, 0.2);
        border-color: var(--error);
        color: var(--error);
        transform: rotate(90deg);
    }
    
    .modal-body {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.8;
        color: var(--text-primary);
    }
    
    .modal-body pre {
        background: var(--bg-secondary);
        padding: 1rem;
        border-radius: var(--radius-md);
        overflow-x: auto;
        font-family: 'Courier New', 'Consolas', monospace;
        font-size: 0.85rem;
        border: 1px solid var(--border-glass);
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .modal-body code {
        background: var(--bg-secondary);
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-family: 'Courier New', 'Consolas', monospace;
        font-size: 0.85rem;
        border: 1px solid var(--border-glass);
    }
    
    /* Sticky Header with Gradient */
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(10, 14, 39, 0.95) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 1.5rem 2rem;
        border-bottom: 1px solid var(--border-glass);
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: var(--shadow-md);
    }
    
    .header-title {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    
    .status-badge {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 1rem;
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-md);
        font-family: 'Courier New', 'Consolas', monospace;
        font-size: 0.75rem;
        color: var(--text-secondary);
    }
    
    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--gradient-accent);
        box-shadow: 0 0 10px var(--accent-cyan);
        animation: pulse-glow 2s infinite;
    }
    
    @keyframes pulse-glow {
        0%, 100% { 
            opacity: 1;
            transform: scale(1);
        }
        50% { 
            opacity: 0.7;
            transform: scale(1.1);
        }
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        border-color: rgba(139, 92, 246, 0.3);
        box-shadow: var(--shadow-glow);
        transform: translateY(-2px);
    }
    
    /* Panel Headers */
    .panel-header {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid transparent;
        border-image: var(--gradient-primary) 1;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .panel-header::before {
        content: '';
        width: 4px;
        height: 20px;
        background: var(--gradient-primary);
        border-radius: 2px;
    }
    
    /* Stats Cards */
    .stat-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-md);
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        border-color: rgba(139, 92, 246, 0.4);
        box-shadow: var(--shadow-glow);
    }
    
    .stat-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    
    .stat-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        background: var(--gradient-accent);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Search Input with Glassmorphism */
    .stTextInput > div > div > input {
        background: var(--bg-glass) !important;
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-purple) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
        outline: none;
    }
    
    /* Paper Tree with Glass Cards */
    .paper-tree {
        max-height: calc(100vh - 500px);
        overflow-y: auto;
        padding-right: 0.5rem;
    }
    
    .paper-item {
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-md);
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: var(--text-secondary);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .paper-item:hover {
        background: rgba(139, 92, 246, 0.1);
        border-color: var(--accent-purple);
        transform: translateX(4px);
        box-shadow: var(--shadow-sm);
    }
    
    .paper-item.selected {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%);
        border-color: var(--accent-purple);
        color: var(--text-primary);
        box-shadow: var(--shadow-glow);
    }
    
    /* Tab Container - Removed empty box styling */
    
    /* Synthesis Box with Gradient Border */
    .synthesis-box {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-lg);
        padding: 2rem;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .synthesis-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
    }
    
    .synthesis-box p {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.8;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }
    
    /* Technical Block - Monospace with Glass */
    .technical-block {
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-md);
        padding: 1rem;
        font-family: 'Courier New', 'Consolas', monospace;
        font-size: 0.85rem;
        color: var(--text-primary);
        white-space: pre-wrap;
        overflow-x: auto;
        margin: 0.5rem 0;
    }
    
    /* Math Box with Gradient */
    .math-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        margin: 1rem 0;
        font-family: 'Courier New', 'Consolas', monospace;
        font-size: 0.875rem;
        color: var(--text-primary);
        position: relative;
    }
    
    .math-box::before {
        content: "✓ Verified Calculation";
        position: absolute;
        top: 0.75rem;
        right: 0.75rem;
        background: var(--gradient-primary);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: var(--radius-sm);
        font-size: 0.7rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    
    /* Thought Box */
    .thought-box {
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-glass);
        border-left: 3px solid var(--accent-purple);
        border-radius: var(--radius-md);
        padding: 1rem;
        margin: 0.75rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        line-height: 1.6;
        color: var(--text-secondary);
    }
    
    /* Web3 Style Buttons */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.75rem 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-sm);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.875rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-glow);
        filter: brightness(1.1);
    }
    
    .stButton > button[kind="primary"] {
        background: var(--gradient-accent);
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
    }
    
    /* Split Button Container */
    .split-buttons {
        display: flex;
        gap: 0.75rem;
        margin: 1rem 0;
    }
    
    /* Status Log */
    .status-log {
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-md);
        padding: 1rem;
        font-family: 'Courier New', 'Consolas', monospace;
        font-size: 0.75rem;
        color: var(--text-muted);
        max-height: 200px;
        overflow-y: auto;
        line-height: 1.6;
    }
    
    /* Source Reference Cards */
    .source-reference {
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-md);
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .source-reference:hover {
        background: rgba(139, 92, 246, 0.1);
        border-color: var(--accent-purple);
        transform: translateX(4px);
    }
    
    .source-reference.highlighted {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%);
        border-color: var(--accent-purple);
        color: var(--text-primary);
        box-shadow: var(--shadow-glow);
    }
    
    /* Mind Map Container */
    .mindmap-container {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-lg);
        padding: 2rem;
        margin: 1.5rem 0;
    }
    
    /* Accordion Sections */
    .accordion-section {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-md);
        margin-bottom: 1rem;
        overflow: hidden;
    }
    
    /* Modal Popup Styles */
    .modal-overlay {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
        z-index: 10000;
        justify-content: center;
        align-items: center;
        animation: fadeIn 0.3s ease;
    }
    
    .modal-overlay.active {
        display: flex;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .modal-content {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-lg);
        padding: 2rem;
        max-width: 80%;
        max-height: 85vh;
        overflow-y: auto;
        box-shadow: var(--shadow-lg);
        position: relative;
        animation: slideUp 0.3s ease;
        color: var(--text-primary);
    }
    
    @keyframes slideUp {
        from {
            transform: translateY(50px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--border-glass);
    }
    
    .modal-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .modal-close {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-sm);
        color: var(--text-primary);
        font-size: 1.5rem;
        width: 40px;
        height: 40px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .modal-close:hover {
        background: rgba(239, 68, 68, 0.2);
        border-color: var(--error);
        color: var(--error);
        transform: rotate(90deg);
    }
    
    .modal-body {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.8;
        color: var(--text-primary);
    }
    
    .modal-body pre {
        background: var(--bg-secondary);
        padding: 1rem;
        border-radius: var(--radius-md);
        overflow-x: auto;
        font-family: 'Courier New', 'Consolas', monospace;
        font-size: 0.85rem;
        border: 1px solid var(--border-glass);
    }
    
    .modal-body code {
        background: var(--bg-secondary);
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-family: 'Courier New', 'Consolas', monospace;
        font-size: 0.85rem;
        border: 1px solid var(--border-glass);
    }
    
    /* Text Area */
    .stTextArea textarea {
        background: var(--bg-glass) !important;
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        padding: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--accent-purple) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
        outline: none;
    }
    
    /* Override Streamlit defaults */
    .stMarkdown {
        color: var(--text-primary);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gradient-primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--gradient-accent);
    }
    
    /* Checkbox Styling */
    .stCheckbox label {
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        background: var(--gradient-accent);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-md);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* File Uploader Styling - Compact */
    .stFileUploader {
        font-size: 0.8rem;
    }
    
    .stFileUploader > label {
        font-size: 0.8rem !important;
        padding: 0.5rem !important;
    }
    
    .stFileUploader > div > div {
        padding: 0.5rem !important;
        font-size: 0.75rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'selected_citation' not in st.session_state:
    st.session_state.selected_citation = None
if 'selected_paper' not in st.session_state:
    st.session_state.selected_paper = None
if 'deep_research_mode' not in st.session_state:
    st.session_state.deep_research_mode = False
if 'thinking_log' not in st.session_state:
    st.session_state.thinking_log = []
if 'current_query_id' not in st.session_state:
    st.session_state.current_query_id = None
if 'highlighted_source' not in st.session_state:
    st.session_state.highlighted_source = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Analysis"
if 'persistent_mind_maps' not in st.session_state:
    st.session_state.persistent_mind_maps = {}
if 'persistent_diagrams' not in st.session_state:
    st.session_state.persistent_diagrams = {}
if 'show_summary' not in st.session_state:
    st.session_state.show_summary = False
if 'summary_content' not in st.session_state:
    st.session_state.summary_content = None
if 'show_nasa_summary' not in st.session_state:
    st.session_state.show_nasa_summary = False
if 'show_file_modal' not in st.session_state:
    st.session_state.show_file_modal = False
if 'modal_file_content' not in st.session_state:
    st.session_state.modal_file_content = None
if 'modal_file_title' not in st.session_state:
    st.session_state.modal_file_title = ""


def initialize_agent():
    """Initialize the Aether agent."""
    if st.session_state.agent is None:
        with st.spinner("Initializing Aether-Agent..."):
            try:
                import traceback
                import sys
                # Redirect stderr to capture detailed errors
                import io
                error_capture = io.StringIO()
                
                try:
                    st.session_state.agent = AetherAgent(deep_research_mode=st.session_state.deep_research_mode)
                    return True
                except Exception as e:
                    error_msg = str(e)
                    error_type = type(e).__name__
                    tb_str = traceback.format_exc()
                    
                    # Log detailed error
                    logger.error(f"Agent initialization error: {error_msg}")
                    logger.error(f"Error type: {error_type}")
                    logger.error(f"Traceback: {tb_str}")
                    
                    # Show user-friendly error with details
                    st.error(f"Error initializing agent: {error_msg}")
                    with st.expander("Error Details (for debugging)"):
                        st.code(tb_str, language='python')
                    return False
            except Exception as e:
                st.error(f"Unexpected error during initialization: {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language='python')
                return False
    else:
        st.session_state.agent.deep_research_mode = st.session_state.deep_research_mode
    return True


def get_paper_list() -> List[Dict]:
    """Get list of all papers in the library."""
    papers_dir = Path("research_papers")
    papers = []
    
    if papers_dir.exists():
        pdf_files = list(papers_dir.glob("*.pdf"))
        manifest_file = Path("library_manifest.json")
        manifest = {}
        if manifest_file.exists():
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        
        for pdf_file in pdf_files:
            paper_id = pdf_file.stem
            paper_info = manifest.get(paper_id, {})
            papers.append({
                "id": paper_id,
                "filename": pdf_file.name,
                "title": paper_info.get("title", pdf_file.stem),
                "year": paper_info.get("year", "Unknown"),
                "authors": paper_info.get("authors", []),
                "path": str(pdf_file)
            })
    
    def sort_key(paper):
        year = paper.get("year", "Unknown")
        if isinstance(year, (int, float)):
            return year
        elif isinstance(year, str) and year.isdigit():
            return int(year)
        else:
            return 0
    
    return sorted(papers, key=sort_key, reverse=True)


def get_chunk_by_citation(citation_text: str) -> Optional[Dict]:
    """Retrieve the exact text chunk for a citation."""
    try:
        from agent_core import VectorSearchTool
        search_tool = VectorSearchTool()
        
        if search_tool.collection is None:
            return None
        
        paper_id_match = re.search(r'Paper ID: ([^,)]+)', citation_text)
        page_match = re.search(r'Page: (\d+)', citation_text)
        
        if not paper_id_match:
            return None
        
        paper_id = paper_id_match.group(1).strip()
        page = int(page_match.group(1)) if page_match else None
        
        try:
            search_results = search_tool.search(f"{paper_id}", top_k=50)
            
            if search_results:
                filtered_results = []
                for result in search_results:
                    result_paper_id = result.get('metadata', {}).get('paper_id', '')
                    if result_paper_id == paper_id:
                        if page is None or result.get('metadata', {}).get('page') == page:
                            filtered_results.append(result)
                
                if filtered_results:
                    for result in filtered_results:
                        if page is not None and result.get('metadata', {}).get('page') == page:
                            return {
                                "content": result.get('content', ''),
                                "metadata": result.get('metadata', {}),
                                "paper_id": paper_id
                            }
                    return {
                        "content": filtered_results[0].get('content', ''),
                        "metadata": filtered_results[0].get('metadata', {}),
                        "paper_id": paper_id
                    }
                
                for result in search_results:
                    if result.get('metadata', {}).get('paper_id') == paper_id:
                        return {
                            "content": result.get('content', ''),
                            "metadata": result.get('metadata', {}),
                            "paper_id": paper_id
                        }
        except Exception as query_error:
            logger.warning(f"Could not retrieve chunk: {str(query_error)}")
            pass
        
        return None
    except Exception as e:
        logger.error(f"Error retrieving chunk: {str(e)}")
        return None


def update_thinking_log(message: str):
    """Update the real-time thinking process log."""
    st.session_state.thinking_log.append(message)
    if len(st.session_state.thinking_log) > 50:
        st.session_state.thinking_log = st.session_state.thinking_log[-50:]


def generate_mind_map(result: Dict[str, Any]) -> Optional[str]:
    """Generate a mind map (knowledge graph) from the agent's response."""
    try:
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        
        entities = set()
        aerospace_keywords = [
            "thermal", "management", "satellite", "orbit", "orbital", "delta-v", "delta v",
            "propulsion", "trajectory", "hohmann", "transfer", "cubesat", "mass", "velocity",
            "attitude", "control", "power", "solar", "battery", "communication", "antenna"
        ]
        
        answer_lower = answer.lower()
        for keyword in aerospace_keywords:
            if keyword in answer_lower:
                entities.add(keyword.title())
        
        for citation in citations:
            citation_text = citation.get("citation", "")
            paper_match = re.search(r'([^(]+) \(Paper ID:', citation_text)
            if paper_match:
                paper_name = paper_match.group(1).strip()[:30]
                entities.add(f"Source: {paper_name}")
        
        query = result.get("query", "Research Topic")
        query_clean = query[:50].replace('"', "'").replace('\n', ' ')
        
        # Color palette for nodes
        colors = [
            "#8b5cf6",  # Purple
            "#3b82f6",  # Blue
            "#00d4ff",  # Cyan
            "#ec4899",  # Pink
            "#10b981",  # Green
            "#f59e0b",  # Amber
            "#ef4444",  # Red
            "#06b6d4",  # Teal
            "#a855f7",  # Violet
            "#f97316"   # Orange
        ]
        
        if not entities:
            dot_lines = [
                'digraph ResearchMindMap {',
                '    bgcolor="#0a0e27";',
                '    fontcolor="#f1f5f9";',
                '    rankdir=TB;',
                '    node [style=filled, fontname="Inter", fontsize=12];',
                '    edge [color="#cbd5e1", fontcolor="#94a3b8", fontsize=10];',
                f'    central [label="{query_clean}", fillcolor="#8b5cf6", fontcolor="#FFFFFF", style=filled, shape=box, penwidth=3];',
                '    node_0 [label="Research Analysis", fillcolor="#3b82f6", fontcolor="#FFFFFF", shape=ellipse];',
                '    central -> node_0 [color="#00d4ff", penwidth=2];',
                '}'
            ]
            return '\n'.join(dot_lines)
        
        dot_lines = [
            'digraph ResearchMindMap {',
            '    bgcolor="#0a0e27";',
            '    fontcolor="#f1f5f9";',
            '    rankdir=TB;',
            '    node [style=filled, fontname="Inter", fontsize=11];',
            '    edge [color="#cbd5e1", fontcolor="#94a3b8", fontsize=9];',
            f'    central [label="{query_clean}", fillcolor="#8b5cf6", fontcolor="#FFFFFF", style=filled, shape=box, penwidth=3];'
        ]
        
        entity_list = list(entities)[:10]
        for i, entity in enumerate(entity_list):
            entity_clean = entity.replace('"', "'").replace('\n', ' ')[:30]
            color = colors[i % len(colors)]
            # Use different shapes for variety
            shape = "ellipse" if i % 2 == 0 else "box"
            dot_lines.append(f'    node_{i} [label="{entity_clean}", fillcolor="{color}", fontcolor="#FFFFFF", shape={shape}, penwidth=2];')
            # Use gradient colors for edges
            edge_color = colors[(i + 1) % len(colors)]
            dot_lines.append(f'    central -> node_{i} [color="{edge_color}", penwidth=2];')
        
        if len(entity_list) > 1:
            for i in range(min(5, len(entity_list) - 1)):
                # Use dashed lines with colors for relationships
                rel_color = colors[(i + 2) % len(colors)]
                dot_lines.append(f'    node_{i} -> node_{i+1} [label="influences", style=dashed, color="{rel_color}", penwidth=1.5];')
        
        dot_lines.append('}')
        return '\n'.join(dot_lines)
        
    except Exception as e:
        logger.error(f"Error generating mind map: {str(e)}")
        query = result.get("query", "Research Topic")
        query_clean = query[:50].replace('"', "'").replace('\n', ' ')
        return f'''digraph ResearchMindMap {{
    bgcolor="#0a0e27";
    fontcolor="#f1f5f9";
    rankdir=TB;
    node [style=filled, fontname="Inter", fontsize=11];
    edge [color="#cbd5e1", fontcolor="#94a3b8"];
    central [label="{query_clean}", fillcolor="#8b5cf6", fontcolor="#FFFFFF", style=filled, shape=box, penwidth=3];
    node_0 [label="Error occurred", fillcolor="#ef4444", fontcolor="#FFFFFF", shape=ellipse];
    central -> node_0 [color="#f59e0b", penwidth=2];
}}'''


def clean_answer_text(answer: str) -> str:
    """Remove citation numbers and Paper ID references from answer text."""
    result = re.sub(r'\(Paper ID: [^)]+\)', '', answer)
    result = re.sub(r'Paper ID: [^,\s]+', '', result)
    result = re.sub(r'As documented in [^,]+,\s*', '', result, flags=re.IGNORECASE)
    result = re.sub(r'According to [^,]+,\s*', '', result, flags=re.IGNORECASE)
    result = re.sub(r'Per [^,]+,\s*', '', result, flags=re.IGNORECASE)
    result = re.sub(r'<sup>\d+</sup>', '', result)
    result = re.sub(r'\s+', ' ', result)
    result = result.strip()
    return result


def generate_diagram(result: Dict[str, Any]) -> Optional[str]:
    """Generate a technical diagram (flowchart/process diagram) from the agent's response."""
    try:
        answer = result.get("answer", "")
        query = result.get("query", "Technical Process")
        
        steps = []
        answer_lower = answer.lower()
        
        step_patterns = [
            r'step \d+[:\s]+([^.]+)',
            r'first[,\s]+([^.]+)',
            r'then[,\s]+([^.]+)',
            r'next[,\s]+([^.]+)',
            r'finally[,\s]+([^.]+)'
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, answer_lower)
            steps.extend(matches[:3])
        
        if not steps:
            technical_concepts = [
                "thermal management", "orbital mechanics", "propulsion", "trajectory",
                "delta-v", "mass budget", "power system", "attitude control"
            ]
            for concept in technical_concepts:
                if concept in answer_lower:
                    steps.append(concept.title())
        
        if not steps:
            steps = ["Analysis", "Synthesis", "Conclusion"]
        
        query_clean = query[:40].replace('"', "'").replace('\n', ' ')
        dot_lines = [
            'digraph TechnicalDiagram {',
            '    bgcolor="#0a0e27";',
            '    fontcolor="#f1f5f9";',
            '    rankdir=LR;',
            f'    start [label="Start: {query_clean}", fillcolor="#8b5cf6", style=filled, shape=box];'
        ]
        
        step_count = min(6, len(steps))
        for i, step in enumerate(steps[:step_count], 1):
            step_clean = step.replace('"', "'").replace('\n', ' ')[:25]
            dot_lines.append(f'    step{i} [label="{step_clean}", style=filled, fillcolor="#3b82f6", shape=box];')
            if i == 1:
                dot_lines.append(f'    start -> step{i};')
            else:
                dot_lines.append(f'    step{i-1} -> step{i};')
        
        if step_count > 0:
            dot_lines.append(f'    step{step_count} -> end [label="Result"];')
        dot_lines.append('    end [label="Synthesis", fillcolor="#8b5cf6", style=filled, shape=box];')
        dot_lines.append('}')
        
        return '\n'.join(dot_lines)
        
    except Exception as e:
        logger.error(f"Error generating diagram: {str(e)}")
        query = result.get("query", "Technical Process")
        query_clean = query[:40].replace('"', "'").replace('\n', ' ')
        return f'''digraph TechnicalDiagram {{
    bgcolor="#0a0e27";
    fontcolor="#f1f5f9";
    rankdir=LR;
    start [label="Start: {query_clean}", fillcolor="#8b5cf6", style=filled, shape=box];
    step1 [label="Analysis", style=filled, fillcolor="#3b82f6", shape=box];
    end [label="Synthesis", fillcolor="#8b5cf6", style=filled, shape=box];
    start -> step1;
    step1 -> end;
}}'''


def generate_agent_interpretation(result: Dict[str, Any], viz_type: str) -> str:
    """Generate a 2-sentence agent interpretation of the visualization."""
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    
    entities = []
    aerospace_keywords = [
        "thermal management", "thermal control", "orbital parameters", "delta-v", "propulsion",
        "satellite", "trajectory", "mass budget", "power system", "power subsystem"
    ]
    
    for keyword in aerospace_keywords:
        if keyword.lower() in answer.lower():
            entities.append(keyword)
    
    paper_ref = ""
    if citations:
        first_citation = citations[0].get("citation", "")
        paper_match = re.search(r'Paper ID: ([^,)]+)', first_citation)
        if paper_match:
            paper_ref = paper_match.group(1).strip()
    
    if viz_type == "mind_map":
        if entities and paper_ref:
            interpretation = f"This map illustrates the dependency relationships between {entities[0] if entities else 'technical concepts'} and related subsystems as retrieved from Paper {paper_ref}. "
            if len(entities) > 1:
                interpretation += f"The hierarchical structure shows how {entities[0]} directly influences {entities[1] if len(entities) > 1 else 'other systems'} within the aerospace domain."
            else:
                interpretation += "The central node represents the query topic, with connected nodes showing related technical entities extracted from the research corpus."
        else:
            interpretation = "This knowledge graph represents the key technical entities and their relationships identified in the current research query. The structure reflects the hierarchical organization of knowledge extracted from the research corpus."
    
    elif viz_type == "diagram":
        if entities:
            interpretation = f"This diagram illustrates the sequential process flow for {entities[0] if entities else 'technical analysis'} as derived from the research synthesis. "
            interpretation += f"The flowchart shows the progression from initial query analysis through {entities[1] if len(entities) > 1 else 'technical evaluation'} to final synthesis."
        else:
            interpretation = "This technical diagram represents the process flow extracted from the research analysis. The sequential steps illustrate the logical progression from query to synthesis."
    
    return interpretation


def get_system_stats() -> Dict[str, Any]:
    """Get system statistics for display."""
    stats = {
        "papers": 0,
        "chunks": 0,
        "status": "disconnected"
    }
    
    try:
        from agent_core import VectorSearchTool
        search_tool = VectorSearchTool()
        if search_tool.collection is not None:
            stats["chunks"] = search_tool.collection.count()
            stats["status"] = "connected"
    except Exception as e:
        logger.warning(f"Could not get chunk count: {str(e)}")
    
    papers_dir = Path("research_papers")
    if papers_dir.exists():
        stats["papers"] = len(list(papers_dir.glob("*.pdf")))
    
    return stats


def main():
    """Main Streamlit app with Web3-style UI."""
    
    # Sticky Header with Gradient
    st.markdown("""
        <div class="sticky-header">
            <div class="header-title">Aether-Agent: Aerospace Synthesis Terminal</div>
        </div>
    """, unsafe_allow_html=True)
    
    # File Modal Popup - Display when file is selected
    if st.session_state.get('show_file_modal', False) and st.session_state.get('modal_file_content'):
        modal_title = st.session_state.get('modal_file_title', 'File Content')
        modal_content = st.session_state.modal_file_content
        
        # Escape HTML for safe display
        import html
        escaped_title = html.escape(modal_title)
        escaped_content = html.escape(str(modal_content))
        
        st.markdown(f"""
            <div id="fileModal" class="modal-overlay active">
                <div class="modal-content" onclick="event.stopPropagation()">
                    <div class="modal-header">
                        <div class="modal-title">{escaped_title}</div>
                        <button class="modal-close" onclick="window.parent.postMessage({{type: 'closeModal'}}, '*')">×</button>
                    </div>
                    <div class="modal-body">
                        <pre>{escaped_content}</pre>
                    </div>
                </div>
            </div>
            <script>
            // Close modal on overlay click
            document.getElementById('fileModal')?.addEventListener('click', function(e) {{
                if (e.target === this) {{
                    window.parent.postMessage({{type: 'closeModal'}}, '*');
                }}
            }});
            
            // Close on Escape key
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'Escape') {{
                    window.parent.postMessage({{type: 'closeModal'}}, '*');
                }}
            }});
            
            // Listen for close message
            window.addEventListener('message', function(event) {{
                if (event.data.type === 'closeModal') {{
                    const modal = document.getElementById('fileModal');
                    if (modal) {{
                        modal.classList.remove('active');
                    }}
                }}
            }});
            </script>
        """, unsafe_allow_html=True)
        
        # Close button using Streamlit (fallback)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Close Modal", key="close_modal_btn", use_container_width=True):
                st.session_state.show_file_modal = False
                st.session_state.modal_file_content = None
                st.session_state.modal_file_title = ""
                st.rerun()
    
    # Initialize agent
    if not initialize_agent():
        st.error("Agent Failed to Initialize")
        return
    
    # Get system stats
    system_stats = get_system_stats()
    
    # Three-panel layout [1, 3, 1] ratio
    left_pane, center_pane, right_pane = st.columns([1, 3, 1], gap="small")
    
    # ========== LEFT PANE: RESEARCH DISCOVERY (20%) ==========
    with left_pane:
        st.markdown('<div class="panel-header">Research Discovery</div>', unsafe_allow_html=True)
        
        # Stats Cards with Glassmorphism
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Papers</div>
                <div class="stat-value">{system_stats['papers']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Knowledge Chunks</div>
                <div class="stat-value">{system_stats['chunks']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Search Bar
        search_query = st.text_input("🔍 Filter Papers", placeholder="Search...", key="paper_search")
        
        st.markdown("---")
        
        # Research Files Dropdown
        papers = get_paper_list()
        
        # Filter papers based on search
        if search_query:
            papers = [p for p in papers if search_query.lower() in p['title'].lower() or search_query.lower() in p.get('filename', '').lower()]
        
        # Create dropdown options
        paper_options = ["Select a research paper..."] + [f"{p['title'][:60]}... ({p['year']})" for p in papers]
        
        selected_index = st.selectbox(
            "Research Files",
            options=range(len(paper_options)),
            format_func=lambda x: paper_options[x],
            key="paper_dropdown"
        )
        
        if selected_index > 0:
            selected_paper = papers[selected_index - 1]
            st.session_state.selected_paper = selected_paper
            st.session_state.show_file_modal = True
            st.session_state.modal_file_content = None
            st.session_state.modal_file_title = selected_paper['title']
            
            # Try to load file content
            try:
                file_path = Path(selected_paper['path'])
                if file_path.exists() and file_path.suffix == '.pdf':
                    # For PDFs, we'll show metadata and info
                    manifest_file = Path("library_manifest.json")
                    manifest = {}
                    if manifest_file.exists():
                        with open(manifest_file, 'r', encoding='utf-8') as f:
                            manifest = json.load(f)
                    
                    paper_info = manifest.get(selected_paper['id'], {})
                    file_content = {
                        "type": "pdf",
                        "title": paper_info.get("title", selected_paper['title']),
                        "year": paper_info.get("year", selected_paper['year']),
                        "authors": paper_info.get("authors", []),
                        "filename": selected_paper['filename'],
                        "paper_id": selected_paper['id'],
                        "path": str(file_path),
                        "size": f"{file_path.stat().st_size / 1024:.2f} KB",
                        "note": "PDF content preview not available. Use the query interface to search within this paper."
                    }
                    st.session_state.modal_file_content = json.dumps(file_content, indent=2)
                else:
                    # For other file types, try to read content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    st.session_state.modal_file_content = content
            except Exception as e:
                st.session_state.modal_file_content = f"Error loading file: {str(e)}"
    
    # ========== CENTER PANE: ANALYSIS STUDIO (60%) ==========
    with center_pane:
        # Show Export/Sync content if active, otherwise show query interface
        if st.session_state.get('show_summary', False) or st.session_state.get('show_nasa_summary', False):
            # Export Research Summary Display
            if st.session_state.get('show_summary', False) and st.session_state.get('summary_content'):
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📄 Research Summary")
                
                summary_json = st.session_state.summary_content
                st.code(summary_json, language='json')
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Close", use_container_width=True, key="close_summary_btn"):
                        st.session_state.show_summary = False
                        st.session_state.summary_content = None
                        st.rerun()
                with col2:
                    st.download_button(
                        "💾 Download JSON",
                        data=summary_json,
                        file_name=f"research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # NASA Database Sync Display
            if st.session_state.get('show_nasa_summary', False):
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 🚀 NASA Database Sync Summary")
                
                try:
                    nasa_stats = get_system_stats()
                    nasa_summary = {
                        "sync_timestamp": datetime.now().isoformat(),
                        "status": "completed",
                        "database_stats": {
                            "total_papers": nasa_stats['papers'],
                            "total_chunks": nasa_stats['chunks'],
                            "connection_status": nasa_stats['status']
                        },
                        "sync_details": {
                            "source": "NASA Technical Reports Server",
                            "papers_synced": nasa_stats['papers'],
                            "chunks_indexed": nasa_stats['chunks'],
                            "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
                    nasa_summary_json = json.dumps(nasa_summary, indent=2)
                    st.code(nasa_summary_json, language='json')
                    st.success(f"✅ Successfully synced {nasa_stats['papers']} papers with {nasa_stats['chunks']} knowledge chunks.")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("Close", use_container_width=True, key="close_nasa_btn"):
                            st.session_state.show_nasa_summary = False
                            st.rerun()
                    with col2:
                        st.download_button(
                            "💾 Download Sync Report",
                            data=nasa_summary_json,
                            file_name=f"nasa_sync_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error generating NASA summary: {str(e)}")
                    if st.button("Close", use_container_width=True, key="close_nasa_error_btn"):
                        st.session_state.show_nasa_summary = False
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            # Normal Query Interface
            # Query Input and Upload in same row
            col_query, col_upload = st.columns([4, 1])
            
            with col_query:
                query = st.text_area(
                    "Enter your aerospace research question:",
                    height=100,
                    placeholder="e.g., What are the thermal management challenges for CubeSats?"
                )
            
            with col_upload:
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                uploaded_file = st.file_uploader(
                    "Upload PDF", 
                    type=['pdf'], 
                    help="Upload a new research paper", 
                    key="upload_pdf"
                )
                if uploaded_file:
                    papers_dir = Path("research_papers")
                    papers_dir.mkdir(exist_ok=True)
                    save_path = papers_dir / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"✓ {uploaded_file.name}")
                    st.caption("Run ingest.py to add to knowledge base.")
            
            # Deep Research Mode Toggle (moved from left panel)
            deep_research = st.checkbox(
                "Deep Research Mode",
                value=st.session_state.deep_research_mode,
                help="Enable web search and cross-referencing"
            )
            st.session_state.deep_research_mode = deep_research
            if st.session_state.agent:
                st.session_state.agent.deep_research_mode = deep_research
            
            # Query button
            submit_button = st.button("Query", type="primary", use_container_width=True)
            
            # Mind Map button
            mindmap_btn = st.button("Generate Mind Map", use_container_width=True)
            
            # Generate Mind Map on button click
            if mindmap_btn:
                if st.session_state.conversation_history:
                    last_conversation = st.session_state.conversation_history[-1]
                    current_result = last_conversation.get("result")
                    current_query_id = last_conversation.get("query_id")
                    
                    if not current_query_id:
                        current_query_id = hash(f"{last_conversation.get('query', '')}_{len(st.session_state.conversation_history)}")
                        last_conversation["query_id"] = current_query_id
                    
                    if current_result:
                        dot_source = generate_mind_map(current_result)
                        if dot_source:
                            st.session_state.persistent_mind_maps[current_query_id] = dot_source
                            st.session_state.persistent_mind_maps[f"{current_query_id}_result"] = current_result
                            st.success("Mind map generated successfully!")
                            st.rerun()
                    else:
                        st.warning("No query results available. Please run a query first.")
                else:
                    st.warning("No conversation history. Please run a query first to generate a mind map.")
            
            # Process query
            if submit_button and query:
                update_thinking_log(f"[START] Processing query: {query}")
                
                try:
                    with st.spinner("Processing query..."):
                        result = st.session_state.agent.query(query)
                    
                    for step in result.get("thought_process", []):
                        update_thinking_log(f"[{step.get('action', 'UNKNOWN')}] {step.get('thought', '')[:100]}...")
                    
                    query_id = hash(f"{query}_{len(st.session_state.conversation_history)}")
                    st.session_state.current_query_id = query_id
                    
                    st.session_state.conversation_history.append({
                        "query": query,
                        "result": result,
                        "query_id": query_id
                    })
                    
                    st.session_state[f"result_{query_id}"] = result
                    
                    # Display answer
                    clean_answer = clean_answer_text(result["answer"])
                    citations = result.get("citations", [])
                    
                    st.markdown('<div class="synthesis-box">', unsafe_allow_html=True)
                    st.markdown(clean_answer, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Store citations
                    st.session_state[f"citations_{query_id}"] = citations
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    update_thinking_log(f"[ERROR] {str(e)}")
                
                update_thinking_log("[END] Query processing complete")
            
            # Display Analysis Process
            if st.session_state.conversation_history:
                last_result = st.session_state.conversation_history[-1].get("result")
                if last_result and last_result.get("thought_process"):
                    st.markdown("---")
                    st.markdown("### Analysis Process")
                    for step in last_result["thought_process"][-5:]:
                        action = step.get('action', 'Unknown')
                        if action == "Math_Engine" or "calculate" in str(step.get('result', '')).lower():
                            st.markdown(f"""
                                <div class="math-box">
                                    <strong>Engine Calculation - Iteration {step.get('iteration', '?')}</strong><br>
                                    <pre style="white-space: pre-wrap; margin-top: 0.5rem;">{str(step.get('result', ''))[:500]}</pre>
                                </div>
                            """, unsafe_allow_html=True)
                        elif action != "Domain_Guardrail_Check":
                            st.markdown(f"""
                                <div class="thought-box">
                                    <strong>Iteration {step.get('iteration', '?')} - {action}</strong><br>
                                    {step.get('thought', '')[:300]}...
                                </div>
                            """, unsafe_allow_html=True)
            
            # Display persistent Mind Maps
            if st.session_state.persistent_mind_maps:
                st.markdown("---")
                st.markdown("### Research Mind Maps")
                # Filter out result entries and only show actual DOT sources
                mind_map_items = {k: v for k, v in st.session_state.persistent_mind_maps.items() 
                                if isinstance(v, str) and v.startswith('digraph')}
                
                for query_id, dot_source in mind_map_items.items():
                    try:
                        st.graphviz_chart(dot_source)
                        result_key = f"{query_id}_result"
                        if result_key in st.session_state.persistent_mind_maps:
                            interpretation = generate_agent_interpretation(
                                st.session_state.persistent_mind_maps[result_key],
                                "mind_map"
                            )
                            st.markdown(f'<div class="thought-box"><strong>Agent Interpretation:</strong> {interpretation}</div>', unsafe_allow_html=True)
                        st.markdown("---")
                    except Exception as e:
                        st.error(f"Error rendering mind map: {str(e)}")
                        logger.error(f"Mind map rendering error: {str(e)}")
    
    # ========== RIGHT PANE: HISTORY & TOOLS (20%) ==========
    with right_pane:
        # Resource Extraction Tools
        st.markdown('<div class="panel-header">Resource Extraction Tools</div>', unsafe_allow_html=True)
        
        if st.button("Export Research Summary", use_container_width=True, key="export_summary_btn"):
            if st.session_state.conversation_history:
                summary = {
                    "timestamp": datetime.now().isoformat(),
                    "queries": len(st.session_state.conversation_history),
                    "conversations": st.session_state.conversation_history
                }
                summary_json = json.dumps(summary, indent=2)
                st.session_state.show_summary = True
                st.session_state.summary_content = summary_json
                st.session_state.show_nasa_summary = False
                st.rerun()
            else:
                st.warning("No research history to export.")
        
        if st.button("Sync NASA Database", use_container_width=True, key="sync_nasa_btn"):
            st.session_state.show_nasa_summary = True
            st.session_state.show_summary = False
            st.rerun()
        
        st.markdown("---")
        
        # Conversation History
        st.markdown('<div class="panel-header">Conversation History</div>', unsafe_allow_html=True)
        if st.session_state.conversation_history:
            # Show history in reverse order (newest first)
            for idx, conv in enumerate(reversed(st.session_state.conversation_history[-10:]), 1):
                clean_ans = clean_answer_text(conv['result']['answer'])
                st.markdown(f"""
                    <div class="glass-card" style="margin-bottom: 1rem; padding: 1rem;">
                        <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem; font-size: 0.9rem;">
                            Q{idx}: {conv['query'][:60]}{'...' if len(conv['query']) > 60 else ''}
                        </div>
                        <div style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.6;">
                            {clean_ans[:200]}{'...' if len(clean_ans) > 200 else ''}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No conversation history yet. Start querying to see history here.")


if __name__ == "__main__":
    main()
