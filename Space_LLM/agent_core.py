"""
Aether-Agent: Agentic Reasoning Core
Implements a ReAct (Reasoning + Acting) agent with tools for aerospace research.
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

# Disable tqdm early to avoid Windows stderr issues in Streamlit
os.environ['TQDM_DISABLE'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# Patch tqdm before any imports that might use it
try:
    import sys
    from io import StringIO
    
    class NullWriter:
        """Null writer to suppress tqdm output."""
        def write(self, s):
            pass
        def flush(self):
            pass
        def isatty(self):
            return False
        def fileno(self):
            return -1
    
    # Store original stderr
    _original_stderr = sys.stderr
except:
    pass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import arxiv
from research_fetcher import download_paper, load_manifest, save_manifest

# Set environment variable for SentenceTransformer cache on Windows
# This helps avoid path length issues
if os.name == 'nt':  # Windows
    cache_home = os.path.expanduser('~/.cache')
    os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME', cache_home)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
# Use absolute path for Windows compatibility
# Handle paths with spaces or special characters that might cause issues
try:
    _chroma_db_path = Path("./chroma_db").resolve()
    # On Windows, check if path has spaces or is too long
    if os.name == 'nt':
        path_str = str(_chroma_db_path)
        # If path has spaces or is too long, use a shorter alternative in user directory
        if ' ' in path_str or len(path_str) > 200:
            # Use user's AppData Local directory as fallback (no spaces, shorter)
            import tempfile
            appdata_local = os.getenv('LOCALAPPDATA', tempfile.gettempdir())
            _chroma_db_path = Path(appdata_local) / "Space_LLM" / "chroma_db"
            logger.info(f"Using alternative path (no spaces): {_chroma_db_path}")
    _chroma_db_path.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_PATH = str(_chroma_db_path)
    logger.info(f"ChromaDB path set to: {CHROMA_DB_PATH}")
except Exception as e:
    logger.warning(f"Error setting up chroma_db path: {e}, using fallback")
    import tempfile
    appdata_local = os.getenv('LOCALAPPDATA', tempfile.gettempdir())
    _chroma_db_path = Path(appdata_local) / "Space_LLM" / "chroma_db"
    _chroma_db_path.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_PATH = str(_chroma_db_path)
    logger.info(f"Using fallback ChromaDB path: {CHROMA_DB_PATH}")

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RESEARCH_PAPERS_DIR = Path("research_papers")
RESEARCH_PAPERS_DIR.mkdir(exist_ok=True)

# System Prompt with Hard-Coded Hallucination Guardrails
SYSTEM_PROMPT = """You are Aether-Agent, an autonomous Aerospace Research Agent for technical synthesis and orbital mechanics.

FORMAL ACADEMIC PROTOCOL (MANDATORY):
- Use third-person academic tone throughout (e.g., "The data indicates..." not "I found...")
- Use superscript-style citations (e.g., "Per Source¹, the Delta-V...") or integrate naturally (e.g., "In accordance with [Source Name], the thermal limit...")
- Avoid conversational filler ("Great question!", "I'd be happy to help", etc.)
- Proceed directly to technical synthesis without preamble
- Citations must be embedded within sentences using superscript notation or natural integration
- Use formal technical language appropriate for academic research papers
- When generating mind maps, think of the response as a Knowledge Graph with entities (nodes) and relationships (edges)

═══════════════════════════════════════════════════════════════════════════════
CRITICAL DOMAIN GUARDRAIL - MUST BE CHECKED BEFORE ANY THOUGHT PROCESS
═══════════════════════════════════════════════════════════════════════════════

DOMAIN RESTRICTION RULE #1 (MANDATORY FIRST STEP):
Before starting your "Thought" process, you MUST perform a 'Topic Check' to evaluate:
"Is this query within the defined Aerospace domain?"

You ONLY answer questions directly related to:
   - Aerospace Engineering
   - Orbital Mechanics
   - Satellite Technology
   - Space Exploration
   - Space systems and satellites
   - Rocket propulsion
   - Space missions
   - Space research
   - Related NASA/ESA/space agency topics

If the user's query is NOT directly related to these topics (e.g., questions about cooking, general history, celebrity news, or any non-aerospace topic):
   - Output EXACTLY: "INVALID QUERY: This system is restricted to Aerospace Research only."
   - DO NOT start the Thought process
   - DO NOT attempt to use any tools (Search_Library, Paper_Finder, Math_Engine)
   - DO NOT search the database
   - STOP immediately and return only the invalid query message

═══════════════════════════════════════════════════════════════════════════════
EVIDENCE-FIRST ANTI-HALLUCINATION CONSTRAINTS (MANDATORY)
═══════════════════════════════════════════════════════════════════════════════

EVIDENCE-FIRST POLICY:
You are FORBIDDEN from using your internal training data to provide technical specifications (like Delta-V, Mass, ISP, orbital parameters, etc.) if that data is not found in the RAG tool (Search_Library).

REQUIRED EVIDENCE USAGE:
- Integrate citations naturally into the prose (e.g., "As documented in [Paper Title] (Paper ID: [ID], Page: [N]), the thermal limit...")
- Every technical specification MUST be attributed to a retrieved source within the sentence
- Use formal citation format: "[Source Title] (Paper ID: [ID], Page: [N])" embedded in text
- DO NOT create separate citation blocks or lists

NO EVIDENCE HANDLING:
- If no evidence is found in the database, state: "No verified sources were located in the database to address this technical question."
- DO NOT infer, estimate, or use general knowledge from training data
- DO NOT provide technical specifications without source attribution
- Maintain formal academic tone even when reporting absence of data

═══════════════════════════════════════════════════════════════════════════════
CRITICAL ANTI-HALLUCINATION RULES (MANDATORY):
═══════════════════════════════════════════════════════════════════════════════

2. DEEP RESEARCH REQUIREMENT:
   - Search with MULTIPLE query variations (at least 3-5 different phrasings)
   - Retrieve AT LEAST 20-30 document chunks (not just 5)
   - Use iterative refinement: search → analyze → search again with refined terms
   - Cross-reference multiple sources before answering
   - Only answer if information appears in MULTIPLE retrieved documents

3. THOUGHT BEFORE ACTION: Every reasoning step must follow this pattern:
   - First: "Thought: [Your reasoning about what you know and what you need]"
   - Then: "Action: [Which tool to use: Search_Library, Paper_Finder, or Math_Engine]"
   - Finally: "Observation: [Result from the tool]"

4. MANDATORY CITATIONS: Every factual claim MUST include:
   - Paper ID (from library_manifest.json or arXiv ID)
   - Page number or section reference
   - Format: "Citation: [Paper Title] (Paper ID: [ID], Page: [N])"
   - Minimum 2-3 citations per answer

5. ZERO HALLUCINATION POLICY:
   - DO NOT make up information
   - DO NOT infer beyond what's explicitly stated in documents
   - DO NOT use general knowledge not found in the corpus
   - If information is not in retrieved documents, state: "Technical data currently unavailable in current aerospace corpus."

6. PHYSICS OVER CHAT: For any mathematical calculation:
   - DO NOT estimate or approximate numbers
   - MUST use Math_Engine tool with actual formulas from papers
   - Extract variables from Search_Library results before calculating
   - Show the formula and all input values before computing

7. GROUNDING REQUIREMENT: Before answering any question:
   - Search_Library MUST be called MULTIPLE times with different queries
   - Retrieve 20-30 chunks minimum
   - Verify answer appears in at least 2-3 different documents
   - If insufficient data, use Paper_Finder
   - Only use Math_Engine with variables extracted from Search_Library results
   - Never calculate with assumed or estimated values

8. ANSWER VALIDATION: Before finalizing any answer:
   - Verify every claim appears in retrieved documents
   - Verify all numbers come from papers (with citations)
   - Verify all formulas are from retrieved content (with citations)
   - If ANY part cannot be verified, state uncertainty clearly
   - Prefer "I don't have enough data" over guessing

Remember: "Technical data currently unavailable" for insufficient data, or "invalid query" for non-space topics is ALWAYS better than hallucinated information."""



class VectorSearchTool:
    """Tool for semantic search in the knowledge base."""
    
    def __init__(self):
        # Initialize ChromaDB first
        try:
            logger.info(f"Initializing ChromaDB at: {CHROMA_DB_PATH}")
            # Ensure path is normalized for Windows and use forward slashes
            db_path = os.path.normpath(CHROMA_DB_PATH).replace('\\', '/')
            # Ensure directory exists
            Path(db_path).mkdir(parents=True, exist_ok=True)
            
            # Try creating client with different path formats if needed
            try:
                self.client = chromadb.PersistentClient(path=db_path)
            except Exception as path_error:
                # Fallback: try with backslashes on Windows
                if os.name == 'nt':
                    db_path_alt = os.path.normpath(CHROMA_DB_PATH)
                    logger.info(f"Trying alternative path format: {db_path_alt}")
                    self.client = chromadb.PersistentClient(path=db_path_alt)
                else:
                    raise path_error
            
            logger.info("ChromaDB client created successfully")
            try:
                self.collection = self.client.get_collection("aerospace_knowledge")
                logger.info("Collection 'aerospace_knowledge' found")
            except Exception as e:
                # Collection doesn't exist yet - will be created during ingestion
                logger.info(f"Collection 'aerospace_knowledge' not found (will be created during ingestion): {str(e)}")
                self.collection = None
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Path attempted: {CHROMA_DB_PATH}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Initialize SentenceTransformer model
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            # Use cache_folder parameter to control where model is cached
            cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'sentence_transformers')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Temporarily redirect stderr to avoid tqdm issues with Streamlit on Windows
            import sys
            old_stderr = sys.stderr
            try:
                # Use NullWriter to suppress tqdm output (defined at module level)
                sys.stderr = NullWriter()
                self.embedding_model = SentenceTransformer(
                    EMBEDDING_MODEL, 
                    cache_folder=cache_dir
                )
            finally:
                sys.stderr = old_stderr
            
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def search(self, query: str, top_k: int = 30) -> List[Dict]:
        """
        Deep search the knowledge base for relevant chunks.
        Uses larger top_k for comprehensive retrieval.
        """
        if self.collection is None:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search ChromaDB with increased results
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, 50)  # Cap at 50 for performance
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted_results
    
    def deep_search(self, query: str) -> List[Dict]:
        """
        Perform deep research with multiple query variations.
        Returns comprehensive results from multiple search angles.
        """
        all_results = []
        seen_ids = set()
        
        # Generate multiple query variations
        query_variations = [
            query,  # Original query
            f"{query} aerospace",  # Add aerospace context
            f"{query} satellite",  # Add satellite context
            f"{query} space",  # Add space context
            query.replace("?", "").replace("what", "how").replace("how", "what"),  # Question variation
        ]
        
        # Search with each variation
        for q_var in query_variations[:3]:  # Use top 3 variations
            results = self.search(q_var, top_k=20)
            for result in results:
                # Use content hash to deduplicate
                content_hash = hash(result['content'][:100])
                if content_hash not in seen_ids:
                    seen_ids.add(content_hash)
                    all_results.append(result)
        
        # Sort by distance (relevance) and return top results
        all_results.sort(key=lambda x: x.get('distance', 1.0))
        return all_results[:30]  # Return top 30 unique results
    
    def extract_variables(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Extract numerical variables and constants from search results.
        Looks for common orbital mechanics variables (mu, r1, r2, v1, v2, delta_v, etc.)
        and physical constants.
        """
        variables = {}
        
        # Common orbital mechanics variable patterns
        patterns = {
            'mu': r'\bmu\s*[=:]\s*([0-9.]+(?:e[+-]?[0-9]+)?)',
            'GM': r'\bGM\s*[=:]\s*([0-9.]+(?:e[+-]?[0-9]+)?)',
            'r1': r'\br1\s*[=:]\s*([0-9.]+(?:e[+-]?[0-9]+)?)',
            'r2': r'\br2\s*[=:]\s*([0-9.]+(?:e[+-]?[0-9]+)?)',
            'a1': r'\ba1\s*[=:]\s*([0-9.]+(?:e[+-]?[0-9]+)?)',
            'a2': r'\ba2\s*[=:]\s*([0-9.]+(?:e[+-]?[0-9]+)?)',
            'v1': r'\bv1\s*[=:]\s*([0-9.]+(?:e[+-]?[0-9]+)?)',
            'v2': r'\bv2\s*[=:]\s*([0-9.]+(?:e[+-]?[0-9]+)?)',
            'delta_v': r'\bdelta[-_]?v\s*[=:]\s*([0-9.]+(?:e[+-]?[0-9]+)?)',
            'eccentricity': r'\be(?:ccentricity)?\s*[=:]\s*([0-9.]+(?:e[+-]?[0-9]+)?)',
        }
        
        # Standard physical constants (Earth)
        standard_constants = {
            'mu_earth': 3.986004418e14,  # m^3/s^2 (Earth's gravitational parameter)
            'R_earth': 6371000.0,  # m (Earth radius)
        }
        
        # Add standard constants
        variables.update(standard_constants)
        
        # Extract from content
        for result in results:
            content = result.get('content', '').lower()
            
            for var_name, pattern in patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    try:
                        # Take the first match and convert to float
                        value = float(matches[0])
                        # Convert to SI units if needed (km -> m, etc.)
                        if var_name in ['r1', 'r2', 'a1', 'a2'] and value < 10000:
                            # Likely in km, convert to m
                            value = value * 1000
                        variables[var_name] = value
                    except ValueError:
                        continue
        
        return variables
    
    def __call__(self, query: str, deep: bool = True) -> str:
        """
        Execute search and return formatted results with citations.
        
        Args:
            query: Search query
            deep: If True, perform deep multi-query search
        """
        if deep:
            results = self.deep_search(query)
        else:
            results = self.search(query, top_k=30)
        
        if not results:
            return "Technical data currently unavailable in current aerospace corpus."
        
        response = f"Found {len(results)} relevant document chunks:\n\n"
        for i, result in enumerate(results[:10], 1):  # Show top 10 in response
            paper_id = result['metadata'].get('paper_id', 'unknown')
            page = result['metadata'].get('page', 'unknown')
            source_file = result['metadata'].get('source_file', 'unknown')
            distance = result.get('distance', 'N/A')
            
            response += f"[{i}] Citation: {source_file} (Paper ID: {paper_id}, Page: {page}, Relevance: {distance:.3f})\n"
            response += f"    Content: {result['content'][:400]}...\n\n"
        
        if len(results) > 10:
            response += f"... and {len(results) - 10} more chunks analyzed.\n"
        
        return response


class PaperFinderTool:
    """Tool to find and download new papers from arXiv."""
    
    def __init__(self):
        self.manifest = load_manifest()
    
    def find_and_download(self, topic: str, max_results: int = 5) -> str:
        """Search arXiv and download relevant papers."""
        logger.info(f"Searching arXiv for: {topic}")
        
        try:
            search = arxiv.Search(
                query=topic,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = list(search.results())
            
            if not results:
                return f"No papers found on arXiv for topic: {topic}"
            
            downloaded = []
            for paper in results:
                paper_id = paper.entry_id.split('/')[-1]
                
                # Check if already in library
                if paper_id in self.manifest:
                    continue
                
                # Download paper
                if download_paper(paper, self.manifest):
                    downloaded.append({
                        "title": paper.title,
                        "arxiv_id": paper_id,
                        "year": paper.published.year
                    })
            
            save_manifest(self.manifest)
            
            if downloaded:
                response = f"Downloaded {len(downloaded)} new papers:\n"
                for paper in downloaded:
                    response += f"  - {paper['title'][:60]}... ({paper['year']})\n"
                response += "\nNote: Run ingest.py to add these papers to the knowledge base."
                return response
            else:
                return f"Found papers on arXiv, but they are already in the library or download failed."
                
        except Exception as e:
            return f"Error searching arXiv: {str(e)}"
    
    def __call__(self, topic: str) -> str:
        """Execute paper finding."""
        return self.find_and_download(topic)


class WebSearchTool:
    """Tool for searching the live web using DuckDuckGo for 2026 aerospace news and updates."""
    
    def __init__(self):
        self.available = False
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            self.available = True
        except ImportError:
            logger.warning("duckduckgo-search not available. Install with: pip install duckduckgo-search")
        except Exception as e:
            logger.warning(f"Error initializing DuckDuckGo search: {str(e)}")
    
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search the web for aerospace-related information.
        Returns list of search results with title, snippet, and URL.
        """
        if not self.available:
            return []
        
        try:
            results = []
            search_query = f"{query} aerospace 2026"
            
            # Search DuckDuckGo
            web_results = list(self.ddgs.text(search_query, max_results=max_results))
            
            for result in web_results:
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("body", ""),
                    "url": result.get("href", ""),
                    "source": "web"
                })
            
            logger.info(f"Web search found {len(results)} results for: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []
    
    def __call__(self, query: str, max_results: int = 5) -> str:
        """Execute web search and return formatted results."""
        results = self.search(query, max_results)
        
        if not results:
            return "No web results found for this query."
        
        response = f"Found {len(results)} web search results:\n\n"
        for i, result in enumerate(results, 1):
            response += f"[{i}] {result['title']}\n"
            response += f"    URL: {result['url']}\n"
            response += f"    Snippet: {result['snippet'][:300]}...\n\n"
        
        return response


class MathEngineTool:
    """Python REPL tool for executing orbital mechanics calculations."""
    
    def __init__(self):
        import math
        import numpy as np
        # Create a restricted namespace for safe execution
        restricted_builtins = {
            'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict,
            'float': float, 'int': int, 'len': len, 'list': list, 'max': max,
            'min': min, 'range': range, 'round': round, 'set': set, 'str': str,
            'sum': sum, 'tuple': tuple, 'zip': zip, 'print': print
        }
        self.math = math
        self.np = np
        self.base_namespace = {
            'math': math,
            'np': np,
            'numpy': np,
            '__builtins__': restricted_builtins
        }
    
    def execute(self, code: str, context_variables: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute Python code safely with optional context variables from Search_Library.
        
        Args:
            code: Python code to execute
            context_variables: Dictionary of variables extracted from Search_Library results
                              (e.g., {'mu': 3.986004418e14, 'r1': 7000e3, 'r2': 42000e3})
        """
        try:
            # Remove markdown code blocks if present
            code = code.strip()
            if code.startswith('```python'):
                code = code[9:]
            if code.startswith('```'):
                code = code[3:]
            if code.endswith('```'):
                code = code[:-3]
            code = code.strip()
            
            # Basic safety check - only allow math operations
            dangerous_keywords = ['import', 'exec', 'eval', 'open', 'file', '__']
            if any(keyword in code.lower() for keyword in dangerous_keywords):
                return "Error: Restricted operation detected. Only mathematical calculations are allowed."
            
            # Create namespace with context variables
            namespace = self.base_namespace.copy()
            if context_variables:
                namespace.update(context_variables)
                # Log which variables are being used
                var_list = ', '.join([f"{k}={v}" for k, v in context_variables.items()])
                logger.info(f"Math_Engine using variables from Search_Library: {var_list}")
            
            # Execute code and show the work
            result = eval(code, namespace)
            
            # Format response showing the Python code executed
            response = "**Python Code Executed:**\n```python\n"
            if context_variables:
                # Show variable assignments
                for var_name, var_value in context_variables.items():
                    response += f"{var_name} = {var_value}\n"
            response += f"{code}\n"
            response += f"# Result: {result}\n```\n\n"
            response += f"**Result:** {result}"
            
            return response
            
        except Exception as e:
            return f"**Python Code Executed:**\n```python\n{code}\n```\n\n**Error:** {str(e)}"
    
    def __call__(self, code: str, context_variables: Optional[Dict[str, Any]] = None) -> str:
        """Execute math code with optional context."""
        return self.execute(code, context_variables)


class AetherAgent:
    """ReAct Agent for aerospace research reasoning."""
    
    # Space/aerospace related keywords for topic validation
    SPACE_KEYWORDS = [
        'space', 'aerospace', 'satellite', 'orbit', 'orbital', 'rocket', 'propulsion',
        'nasa', 'esa', 'spacecraft', 'mission', 'launch', 'trajectory', 'delta-v',
        'hohmann', 'transfer', 'cubesat', 'thermal', 'management', 'spacecraft',
        'aerodynamics', 'atmosphere', 'reentry', 'space station', 'iss', 'mars',
        'moon', 'lunar', 'planetary', 'asteroid', 'comet', 'space debris',
        'attitude', 'control', 'gyroscope', 'thruster', 'solar panel', 'battery',
        'communication', 'antenna', 'payload', 'ground station', 'telemetry'
    ]
    
    def __init__(self, llm_provider=None, deep_research_mode: bool = False):
        """
        Initialize the agent with tools.
        llm_provider: Optional LLM provider (OpenAI, Anthropic, etc.)
        If None, uses a simple rule-based reasoning system.
        deep_research_mode: If True, enables web search and cross-referencing with local database.
        """
        try:
            logger.info("Initializing VectorSearchTool...")
            self.search_tool = VectorSearchTool()
            logger.info("Initializing PaperFinderTool...")
            self.paper_finder = PaperFinderTool()
            logger.info("Initializing MathEngineTool...")
            self.math_engine = MathEngineTool()
            logger.info("Initializing WebSearchTool...")
            self.web_search = WebSearchTool()
        except Exception as e:
            logger.error(f"Error initializing agent tools: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        self.thought_process = []
        self.llm_provider = llm_provider
        self.system_prompt = SYSTEM_PROMPT  # Hard-coded hallucination guardrails
        self.deep_research_mode = deep_research_mode
    
    def is_space_related(self, query: str) -> bool:
        """
        Check if query is related to space/aerospace topics.
        Returns True if space-related, False otherwise.
        Strict validation: requires at least one space keyword.
        """
        query_lower = query.lower()
        
        # Check for space keywords - MUST have at least one
        has_space_keyword = any(keyword in query_lower for keyword in self.SPACE_KEYWORDS)
        
        # Check for common non-space topics (negative check)
        non_space_topics = [
            'cooking', 'recipe', 'food', 'restaurant', 'movie', 'film', 'actor',
            'music', 'song', 'sport', 'football', 'basketball', 'game', 'video game',
            'history', 'ancient', 'philosophy', 'religion', 'politics', 'election',
            'cooking recipe', 'how to cook', 'best restaurant', 'movie review',
            'medical', 'medicine', 'health', 'business', 'finance', 'economics',
            'travel', 'tourism', 'art', 'literature'
        ]
        
        # If query has non-space topics without space context, reject
        has_non_space = any(topic in query_lower for topic in non_space_topics)
        if has_non_space and not has_space_keyword:
            return False
        
        # Must have at least one space keyword to be valid
        return has_space_keyword
    
    def generate_research_plan(self, query: str) -> str:
        """
        Generate a research plan for Deep Research mode.
        Returns a structured plan with steps.
        """
        plan = f"**Research Plan for: {query}**\n\n"
        
        steps = [
            "Step 1: Search Local PDF Database",
            "Step 2: Extract Key Technical Specifications",
            "Step 3: Search Web for 2026 Updates and News",
            "Step 4: Cross-reference Web Results with Local Database",
            "Step 5: Identify Contradictions and Gaps",
            "Step 6: Synthesize Findings with Source Attribution"
        ]
        
        for i, step in enumerate(steps, 1):
            plan += f"{step}\n"
        
        plan += "\n**Context Steering:** Prioritizing local PDFs as Primary Truth, web as Secondary Context."
        
        return plan
    
    def apply_context_steering(self, local_results: List[Dict], web_results: List[Dict]) -> Dict[str, Any]:
        """
        Apply Context Steering: Prioritize local PDFs as Primary Truth, web as Secondary Context.
        Returns combined and prioritized results.
        """
        prioritized_results = {
            "primary_sources": [],  # Local PDFs (Primary Truth)
            "secondary_sources": [],  # Web results (Secondary Context)
            "cross_references": []  # Connections between local and web
        }
        
        # Add local results as primary sources
        for result in local_results:
            prioritized_results["primary_sources"].append({
                **result,
                "source_type": "local_pdf",
                "priority": "primary"
            })
        
        # Add web results as secondary sources
        for result in web_results:
            prioritized_results["secondary_sources"].append({
                **result,
                "source_type": "web",
                "priority": "secondary"
            })
        
        # Cross-reference: Find connections between local and web
        for local_result in local_results[:5]:  # Top 5 local results
            local_content = local_result.get('content', '').lower()
            local_keywords = set(local_content.split()[:20])  # Top keywords
            
            for web_result in web_results:
                web_content = (web_result.get('snippet', '') + ' ' + web_result.get('title', '')).lower()
                web_keywords = set(web_content.split()[:20])
                
                # Check for keyword overlap
                overlap = local_keywords.intersection(web_keywords)
                if len(overlap) >= 3:  # At least 3 matching keywords
                    prioritized_results["cross_references"].append({
                        "local_source": local_result.get('metadata', {}).get('source_file', 'unknown'),
                        "web_source": web_result.get('title', 'unknown'),
                        "web_url": web_result.get('url', ''),
                        "common_topics": list(overlap)[:5]
                    })
        
        return prioritized_results
    
    def validate_answer_grounding(self, answer: str, retrieved_chunks: List[Dict]) -> bool:
        """
        Validate that the answer is actually grounded in retrieved documents.
        Returns True if answer appears to be grounded, False otherwise.
        """
        if not retrieved_chunks:
            return False
        
        answer_lower = answer.lower()
        answer_keywords = set(answer_lower.split())
        
        # Check if key terms from answer appear in retrieved chunks
        matches = 0
        for chunk in retrieved_chunks:
            content_lower = chunk.get('content', '').lower()
            # Count matching keywords
            chunk_keywords = set(content_lower.split())
            overlap = answer_keywords.intersection(chunk_keywords)
            if len(overlap) > 5:  # At least 5 matching keywords
                matches += 1
        
        # Answer is grounded if it matches at least 2 chunks
        return matches >= 2
    
    def think(self, observation: str, query: str = "", retrieved_chunks: Optional[List[Dict]] = None) -> str:
        """
        Generate a thought based on observation, following SYSTEM_PROMPT rules.
        Includes domain evaluation as first step.
        """
        # STEP 1: Domain Evaluation (MUST be first)
        thought = "Thought: Domain Check - Is this query within the defined Aerospace domain?\n"
        thought += f"Evaluating query: '{query}'\n"
        
        if not self.is_space_related(query):
            thought += "Domain Check Result: Query is NOT within Aerospace domain. "
            thought += "I must return 'INVALID QUERY: This system is restricted to Aerospace Research only.' "
            thought += "I will NOT use any tools or attempt to answer."
            return thought
        
        thought += "Domain Check Result: Query is within Aerospace domain. Proceeding with evidence-based reasoning.\n\n"
        
        # STEP 2: Evidence-First Evaluation
        if retrieved_chunks is None or len(retrieved_chunks) == 0:
            thought += "Evidence Check: No evidence found in database. "
            thought += "I cannot use internal training data. "
            thought += "I must say: 'I cannot find a verified source in the database to answer this technical question accurately.'"
            return thought
        
        thought += f"Evidence Check: Found {len(retrieved_chunks)} document chunks in database. "
        thought += "I can proceed with evidence-based answer using 'According to [Source]...' format.\n\n"
        
        # STEP 3: Apply reasoning logic following system prompt rules
        if "unavailable" in observation.lower() or "not found" in observation.lower():
            thought += "Reasoning: The knowledge base does not contain sufficient information. "
            thought += "I must trigger Paper_Finder to search for new research papers before answering. "
            thought += "I will NOT make up information or use training data."
        elif any(keyword in query.lower() for keyword in ["calculate", "compute", "solve", "formula", "delta-v", "velocity"]):
            thought += "Reasoning: This query requires mathematical calculation. "
            thought += "I must first extract variables from Search_Library results, then use Math_Engine with those exact values. "
            thought += "I will NOT estimate or approximate - all values must come from retrieved sources."
        else:
            thought += "Reasoning: Found relevant information in the knowledge base. "
            thought += "I will provide an answer using 'According to [Source]...' format with proper citations."
        
        return thought
    
    def act(self, thought: str, query: str, context_variables: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
        """
        Execute an action based on thought.
        
        Args:
            thought: The reasoning thought
            query: The user query
            context_variables: Variables extracted from Search_Library to pass to Math_Engine
        """
        action_type = None
        result = None
        
        # Determine action from thought
        if "Paper_Finder" in thought or "paper" in thought.lower():
            # Extract topic from query or thought
            action_type = "Paper_Finder"
            result = self.paper_finder(query)
            
        elif "Math_Engine" in thought or "calculate" in thought.lower() or "formula" in thought.lower():
            action_type = "Math_Engine"
            # Use context variables from Search_Library if available
            if context_variables:
                # Try to construct code from query and variables
                # For now, pass the query and let Math_Engine handle it with context
                result = self.math_engine(query, context_variables)
            else:
                # Fallback: try to extract code from query
                result = self.math_engine(query)
            
        else:
            action_type = "Search_Library"
            result = self.search_tool(query, deep=True)  # Use deep search by default
        
        return action_type, result
    
    def react_loop(self, query: str, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Execute ReAct reasoning loop following SYSTEM_PROMPT guardrails.
        Returns complete reasoning trace with citations.
        """
        self.thought_process = []
        current_query = query
        iteration = 0
        extracted_variables = None  # Store variables from Search_Library for Math_Engine
        
        # STEP 1: DOMAIN GUARDRAIL - Topic Check BEFORE Thought Process
        # This MUST be the first step before any reasoning
        logger.info(f"Performing Domain Guardrail check for: {query}")
        
        if not self.is_space_related(query):
            logger.info(f"Query not space-related: {query}")
            return {
                "query": query,
                "answer": "INVALID QUERY: This system is restricted to Aerospace Research only.",
                "thought_process": [{
                    "iteration": 0,
                    "thought": "Domain Check - Is this query within the defined Aerospace domain?\n"
                               f"Evaluating query: '{query}'\n"
                               "Domain Check Result: Query is NOT within Aerospace domain. "
                               "I must return 'INVALID QUERY: This system is restricted to Aerospace Research only.' "
                               "I will NOT use any tools or attempt to answer.",
                    "action": "Domain_Guardrail_Check",
                    "result": "Query rejected - not within Aerospace domain. No tools executed."
                }],
                "citations": [],
                "system_prompt": SYSTEM_PROMPT
            }
        
        # Log system prompt at start (only after domain validation passes)
        self.thought_process.append({
            "iteration": 0,
            "thought": "Domain Check - Is this query within the defined Aerospace domain?\n"
                      f"Evaluating query: '{query}'\n"
                      "Domain Check Result: Query is within Aerospace domain. Proceeding with evidence-based reasoning.",
            "action": "Domain_Guardrail_Check",
            "result": "Domain validation passed. Query is within Aerospace domain."
        })
        
        # Deep Research Mode: Generate Research Plan
        web_results = []
        if self.deep_research_mode:
            research_plan = self.generate_research_plan(query)
            self.thought_process.append({
                "iteration": 0.5,
                "thought": "Deep Research Mode Enabled: Generating research plan with web search integration.",
                "action": "Research_Plan_Generation",
                "result": research_plan
            })
            
            # Step 3: Search Web for 2026 updates
            logger.info("Deep Research Mode: Searching web for 2026 aerospace updates")
            web_results_raw = self.web_search.search(query, max_results=5)
            web_results = web_results_raw
        
        while iteration < max_iterations:
            iteration += 1
            
            # Step 1: Deep search library (Rule 7: Deep Research Requirement)
            logger.info(f"Performing deep search for: {current_query}")
            search_results_raw = self.search_tool.deep_search(current_query)
            
            if not search_results_raw:
                # Try regular search as fallback
                search_results_raw = self.search_tool.search(current_query, top_k=30)
            
            search_result = self.search_tool(current_query, deep=True)
            
            logger.info(f"Retrieved {len(search_results_raw)} chunks from deep search")
            
            # Extract variables from search results for Math_Engine
            if search_results_raw:
                extracted_variables = self.search_tool.extract_variables(search_results_raw)
                if extracted_variables:
                    logger.info(f"Extracted variables from Search_Library: {extracted_variables}")
            
            # Deep Research Mode: Apply Context Steering
            if self.deep_research_mode and web_results:
                context_steered = self.apply_context_steering(search_results_raw, web_results)
                self.thought_process.append({
                    "iteration": iteration - 0.3,
                    "thought": "Context Steering: Prioritizing local PDFs as Primary Truth, web as Secondary Context.",
                    "action": "Context_Steering",
                    "result": f"Primary Sources: {len(context_steered['primary_sources'])}, "
                             f"Secondary Sources: {len(context_steered['secondary_sources'])}, "
                             f"Cross-references: {len(context_steered['cross_references'])}"
                })
            
            # Think (following Evidence-First policy - includes domain check and evidence evaluation)
            thought = self.think(search_result, current_query, search_results_raw)
            self.thought_process.append({
                "iteration": iteration,
                "thought": thought,
                "action": "Search_Library",
                "result": search_result,
                "extracted_variables": extracted_variables if extracted_variables else None,
                "chunks": search_results_raw,  # Store chunks for Evidence-First answer compilation
                "web_results": web_results if self.deep_research_mode else []  # Store web results for context steering
            })
            
            # Evidence-First Check: If no evidence found, return appropriate message
            if not search_results_raw or len(search_results_raw) == 0:
                logger.warning("No evidence found in database - cannot use training data")
                return {
                    "query": query,
                    "answer": "No verified sources were located in the database to address this technical question.",
                    "thought_process": self.thought_process,
                    "citations": [],
                    "system_prompt": SYSTEM_PROMPT,
                    "chunks_analyzed": 0
                }
            
            # Check if we have sufficient information (Evidence-First policy)
            if "unavailable" in search_result.lower() or "not found" in search_result.lower():
                # Need to find more papers
                thought = f"Thought: Evidence-First Check - I don't have enough verified data on '{current_query}' in my database. "
                thought += "I will trigger the Paper_Finder to fetch new NASA/arXiv research before answering. "
                thought += "I will NOT make up information or use training data."
                
                action_type, result = self.act(thought, current_query)
                
                self.thought_process.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": action_type,
                    "result": result
                })
                
                # After finding papers, search again
                if "Downloaded" in result:
                    search_results_raw = self.search_tool.search(current_query)
                    search_result = self.search_tool(current_query)
                    # Re-extract variables after new search
                    if search_results_raw:
                        extracted_variables = self.search_tool.extract_variables(search_results_raw)
                    
                    self.thought_process.append({
                        "iteration": iteration + 0.5,
                        "thought": "Re-searching library after adding new papers",
                        "action": "Search_Library",
                        "result": search_result,
                        "extracted_variables": extracted_variables if extracted_variables else None
                    })
            
            # Check if we should use math engine (Rule 4: Physics over Chat)
            if any(keyword in query.lower() for keyword in ["calculate", "compute", "solve", "formula", "velocity", "delta-v", "delta v"]):
                thought = "Thought: Following Rule 4 (Physics over Chat), this query requires mathematical calculation. "
                thought += f"I will use Math_Engine with variables extracted from Search_Library: {extracted_variables if extracted_variables else 'No variables found - will use standard constants'}."
                
                action_type, result = self.act(thought, query, extracted_variables)
                
                self.thought_process.append({
                    "iteration": iteration + 1,
                    "thought": thought,
                    "action": action_type,
                    "result": result,
                    "variables_used": extracted_variables if extracted_variables else None
                })
            
            # Break if we have good results
            if "unavailable" not in search_result.lower() and "not found" not in search_result.lower():
                break
        
        # Compile final answer with citations (Rule 2: Mandatory Citations)
        final_answer = self._compile_answer()
        
        # Validate answer grounding (Evidence-First policy)
        if search_results_raw:
            is_grounded = self.validate_answer_grounding(final_answer, search_results_raw)
            if not is_grounded and "unavailable" not in final_answer.lower() and "cannot find" not in final_answer.lower():
                logger.warning("Answer may not be properly grounded in retrieved documents")
                final_answer = "No verified sources were located in the database to address this technical question."
        
        return {
            "query": query,
            "answer": final_answer,
            "thought_process": self.thought_process,
            "citations": self._extract_citations(),
            "system_prompt": SYSTEM_PROMPT,  # Include system prompt in response
            "chunks_analyzed": len(search_results_raw) if search_results_raw else 0
        }
    
    def _compile_answer(self) -> str:
        """
        Compile final answer from thought process using formal academic format.
        Integrates citations naturally into prose using third-person academic tone.
        """
        # Collect all successful search results with their chunks
        search_results = []
        search_chunks = []
        for step in self.thought_process:
            if step["action"] == "Search_Library" and "unavailable" not in step.get("result", "").lower():
                search_results.append(step["result"])
                # Try to get chunks from the step if available
                if "chunks" in step:
                    search_chunks.extend(step["chunks"])
        
        if not search_results:
            return "No verified sources were located in the database to address this technical question."
        
        # Extract citations
        citations = self._extract_citations()
        if not citations:
            return "No verified sources were located in the database to address this technical question."
        
        # Get chunks for content extraction
        if not search_chunks:
            for step in self.thought_process:
                if "chunks" in step and step["chunks"]:
                    search_chunks.extend(step["chunks"][:10])  # Top 10 chunks
        
        # Build formal academic answer with integrated citations
        formatted_answer = ""
        
        # Extract key information from chunks and integrate citations naturally
        used_citations = set()
        citation_map = {}
        
        # Build citation map from extracted citations
        for citation in citations:
            citation_text = citation.get("citation", "")
            # Parse citation format: "filename.pdf (Paper ID: paper_id, Page: N)"
            paper_match = re.search(r'Paper ID: ([^,)]+)', citation_text)
            page_match = re.search(r'Page: (\d+)', citation_text)
            if paper_match:
                paper_id = paper_match.group(1).strip()
                page = page_match.group(1) if page_match else None
                citation_map[paper_id] = {
                    "citation": citation_text,
                    "page": page,
                    "source_file": citation_text.split("(")[0].strip() if "(" in citation_text else citation_text
                }
        
        # Synthesize answer from chunks with integrated citations
        if search_chunks:
            # Group chunks by paper
            papers_content = {}
            for chunk in search_chunks[:15]:  # Use top 15 chunks
                paper_id = chunk.get('metadata', {}).get('paper_id', 'unknown')
                if paper_id not in papers_content:
                    papers_content[paper_id] = []
                papers_content[paper_id].append(chunk)
            
            # Build answer with integrated citations
            paragraphs = []
            for paper_id, chunks in list(papers_content.items())[:3]:  # Use top 3 papers
                if paper_id in citation_map:
                    citation_info = citation_map[paper_id]
                    citation_ref = f"{citation_info['source_file']} (Paper ID: {paper_id}, Page: {citation_info['page']})"
                    
                    # Extract key content from chunks
                    content_parts = []
                    for chunk in chunks[:3]:  # Top 3 chunks per paper
                        content = chunk.get('content', '').strip()
                        if content and len(content) > 50:
                            content_parts.append(content[:300])  # First 300 chars
                    
                    if content_parts:
                        # Integrate content without citation references in main answer
                        paragraph = content_parts[0]
                        if len(content_parts) > 1:
                            paragraph += f" Additional analysis indicates {content_parts[1][:200]}."
                        paragraphs.append(paragraph)
                        used_citations.add(citation_ref)
            
            # Don't add citation references to main answer text
            
            formatted_answer = " ".join(paragraphs)
        
        # Fallback: use best result without citation references
        if not formatted_answer:
            best_result = max(search_results, key=len)
            # Remove citation markers and clean content
            content = best_result.replace("Citation:", "").replace("[", "").replace("]", "")
            # Remove Paper ID references
            content = re.sub(r'\(Paper ID: [^)]+\)', '', content)
            content = re.sub(r'Paper ID: [^,\s]+', '', content)
            formatted_answer = content[:1000]  # Limit length
        
        # Ensure formal third-person tone (remove first-person references)
        formatted_answer = formatted_answer.replace("I found", "The analysis indicates")
        formatted_answer = formatted_answer.replace("I can", "The data")
        formatted_answer = formatted_answer.replace("I will", "The system will")
        formatted_answer = formatted_answer.replace("I have", "The research has")
        
        return formatted_answer
    
    def _extract_citations(self) -> List[Dict]:
        """Extract citations from thought process."""
        citations = []
        seen = set()
        
        for step in self.thought_process:
            if "Citation:" in step.get("result", ""):
                # Extract citation info
                result = step["result"]
                for line in result.split("\n"):
                    if "Citation:" in line:
                        citation = line.split("Citation:")[-1].strip()
                        if citation not in seen:
                            seen.add(citation)
                            citations.append({"citation": citation})
        
        return citations
    
    def query(self, question: str) -> Dict[str, Any]:
        """Main query interface."""
        return self.react_loop(question)


def main():
    """Test the agent."""
    agent = AetherAgent()
    
    test_queries = [
        "What are the key challenges in thermal management for small satellites?",
        "How do you calculate delta-v for a Hohmann transfer?",
    ]
    
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("=" * 60)
        
        result = agent.query(query)
        
        print("\nAnswer:")
        print(result["answer"])
        
        print("\nCitations:")
        for citation in result["citations"]:
            print(f"  - {citation['citation']}")


if __name__ == "__main__":
    main()
