# Flask UI for Aether-Agent

## Overview

A professional Flask-based web interface for Aether-Agent with a modern Gemini 3-style design. The UI maintains all backend functionality while providing a clean, responsive interface.

## Features

- **Modern Design**: Gemini 3-inspired UI with clean aesthetics
- **Responsive Layout**: Works on desktop, tablet, and mobile devices
- **Real-time Status**: Live knowledge base and paper count
- **Thought Process Visualization**: Expandable thought process display
- **Citation Display**: Formatted citations with paper references
- **Conversation History**: Local storage of recent queries
- **Professional Styling**: Custom CSS with smooth animations

## File Structure

```
Space_LLM/
├── app_flask.py          # Flask application (main entry point)
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Professional styling
│   └── js/
│       └── app.js        # Frontend JavaScript
└── agent_core.py         # Backend agent (unchanged)
```

## Running the Flask App

### Method 1: Direct Python
```bash
python app_flask.py
```

### Method 2: Flask CLI
```bash
flask --app app_flask run
```

### Method 3: With Custom Port
```bash
python app_flask.py
# Or
flask --app app_flask run --port 5000
```

The app will be available at: **http://localhost:5000**

## API Endpoints

### POST `/api/query`
Submit a query to the agent.

**Request:**
```json
{
  "question": "What are thermal management challenges for small satellites?"
}
```

**Response:**
```json
{
  "success": true,
  "answer": "...",
  "citations": [...],
  "thought_process": [...],
  "query": "..."
}
```

### GET `/api/status`
Get system status (chunks, papers count).

**Response:**
```json
{
  "success": true,
  "chunks": 2311,
  "papers": 40,
  "status": "ready"
}
```

### GET `/api/health`
Health check endpoint.

## UI Components

### Sidebar
- System status display
- Knowledge base statistics
- About information

### Main Content
- Query input area
- Answer display
- Citations section
- Thought process (expandable)
- Conversation history

## Design Features

- **Color Scheme**: Google Material Design inspired
- **Typography**: Inter font family
- **Animations**: Smooth transitions and loading indicators
- **Shadows**: Layered elevation system
- **Responsive**: Mobile-first approach

## Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari
- Opera

## Development

### Making Changes

1. **CSS**: Edit `static/css/style.css`
2. **JavaScript**: Edit `static/js/app.js`
3. **HTML**: Edit `templates/index.html`
4. **Backend**: Edit `app_flask.py` (routes) or `agent_core.py` (agent logic)

### Testing

After making changes, refresh your browser. Flask's debug mode will auto-reload on code changes.

## Notes

- Backend functionality (`agent_core.py`) remains unchanged
- Only the UI layer has been replaced (Streamlit → Flask)
- All agent capabilities preserved:
  - Semantic search
  - Paper finding
  - Math engine
  - Citation tracking
  - Thought process logging
