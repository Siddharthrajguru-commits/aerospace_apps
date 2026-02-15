"""
Aether-Agent: Flask Web Application
Professional UI with Gemini 3-style design
"""

from flask import Flask, render_template, request, jsonify, session
from agent_core import AetherAgent
import json
import os
from pathlib import Path

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
app.config['JSON_AS_ASCII'] = False  # Support Unicode in JSON

# Initialize agent (will be lazy-loaded)
agent = None

def get_agent():
    """Lazy load agent to avoid initialization on import."""
    global agent
    if agent is None:
        agent = AetherAgent()
    return agent

@app.route('/')
def index():
    """Main page route."""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    """API endpoint for agent queries."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Please provide a question.'
            }), 400
        
        # Get agent and process query
        agent_instance = get_agent()
        result = agent_instance.query(question)
        
        return jsonify({
            'success': True,
            'answer': result.get('answer', ''),
            'citations': result.get('citations', []),
            'thought_process': result.get('thought_process', []),
            'query': question
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Get system status."""
    try:
        from agent_core import VectorSearchTool
        search_tool = VectorSearchTool()
        
        chunk_count = 0
        if search_tool.collection is not None:
            chunk_count = search_tool.collection.count()
        
        # Count papers
        papers_dir = Path("research_papers")
        paper_count = len(list(papers_dir.glob("*.pdf"))) if papers_dir.exists() else 0
        
        return jsonify({
            'success': True,
            'chunks': chunk_count,
            'papers': paper_count,
            'status': 'ready'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
