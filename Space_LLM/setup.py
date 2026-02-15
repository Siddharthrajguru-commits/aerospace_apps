"""
Aether-Agent: Quick Setup Script
Runs the complete setup pipeline: fetch papers -> ingest -> ready to use
"""

import subprocess
import sys
from pathlib import Path

def run_setup():
    """Run the complete setup process."""
    print("=" * 60)
    print("Aether-Agent: Complete Setup")
    print("=" * 60)
    
    # Step 1: Fetch papers
    print("\n[1/2] Fetching research papers from arXiv...")
    print("-" * 60)
    try:
        subprocess.run([sys.executable, "research_fetcher.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error fetching papers: {e}")
        return False
    
    # Step 2: Ingest papers
    print("\n[2/2] Ingesting papers into knowledge base...")
    print("-" * 60)
    try:
        subprocess.run([sys.executable, "ingest.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error ingesting papers: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run 'streamlit run app.py' to start the web interface")
    print("  - Or run 'python agent_core.py' to test the agent")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = run_setup()
    sys.exit(0 if success else 1)
