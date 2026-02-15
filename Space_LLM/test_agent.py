"""Quick test script for the agent"""
import sys
import io
from agent_core import AetherAgent

# Fix Unicode encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    # Initialize agent
    print("Initializing agent...")
    agent = AetherAgent()
    print("Agent initialized successfully!")

    # Test query
    print("\n" + "="*60)
    print("Testing Query: 'What are thermal management challenges for small satellites?'")
    print("="*60)

    result = agent.query("What are thermal management challenges for small satellites?")

    print("\nAnswer:")
    print(result["answer"])

    print("\nCitations:")
    for citation in result["citations"]:
        print(f"  - {citation.get('citation', 'N/A')}")

    print("\n" + "="*60)
    print("Test completed!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
