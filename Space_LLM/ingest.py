"""
Aether-Agent: Semantic Data Ingestion Module
Processes PDFs into a knowledge base using semantic chunking and ChromaDB.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict

# Disable tqdm early to avoid Windows stderr issues
os.environ['TQDM_DISABLE'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# Patch stderr before any imports that might use tqdm
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
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Set environment variable for SentenceTransformer cache on Windows
if os.name == 'nt':  # Windows
    cache_home = os.path.expanduser('~/.cache')
    os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME', cache_home)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
RESEARCH_PAPERS_DIR = Path("research_papers")
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

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


class KnowledgeIngester:
    """Handles PDF ingestion, chunking, and vector storage."""
    
    def __init__(self):
        """Initialize the ingester with embedding model and ChromaDB."""
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        try:
            # Use cache_folder parameter to control where model is cached
            cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'sentence_transformers')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Temporarily redirect stderr to avoid tqdm issues with Streamlit on Windows
            import sys
            old_stderr = sys.stderr
            try:
                # Use NullWriter to suppress tqdm output
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
        
        # Initialize ChromaDB client with absolute path
        try:
            logger.info(f"Initializing ChromaDB at: {CHROMA_DB_PATH}")
            # Ensure path is normalized for Windows
            db_path = os.path.normpath(CHROMA_DB_PATH)
            # Ensure directory exists
            Path(db_path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info("ChromaDB client created successfully")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="aerospace_knowledge",
            metadata={"description": "Aerospace research papers knowledge base"}
        )
        
        logger.info(f"ChromaDB initialized at: {CHROMA_DB_PATH}")
    
    def load_pdf(self, pdf_path: Path) -> List[Dict]:
        """Load and extract text from a PDF file."""
        logger.info(f"Loading PDF: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata['source_file'] = pdf_path.name
            doc.metadata['paper_id'] = pdf_path.stem
        
        logger.info(f"  Extracted {len(documents)} pages from {pdf_path.name}")
        return documents
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Split documents into semantic chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"  Created {len(chunks)} chunks from {len(documents)} pages")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def store_in_chromadb(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Store chunks and embeddings in ChromaDB."""
        logger.info("Storing chunks in ChromaDB...")
        
        ids = []
        documents = []
        metadatas = []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{chunk.metadata.get('paper_id', 'unknown')}_chunk_{idx}"
            ids.append(chunk_id)
            documents.append(chunk.page_content)
            
            # Prepare metadata
            metadata = {
                "source_file": chunk.metadata.get('source_file', 'unknown'),
                "paper_id": chunk.metadata.get('paper_id', 'unknown'),
                "page": chunk.metadata.get('page', 0),
                "chunk_index": idx
            }
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"✓ Stored {len(chunks)} chunks in ChromaDB")
    
    def ingest_paper(self, pdf_path: Path) -> bool:
        """Complete ingestion pipeline for a single PDF."""
        try:
            # Load PDF
            documents = self.load_pdf(pdf_path)
            
            # Chunk documents
            chunks = self.chunk_documents(documents)
            
            # Generate embeddings
            texts = [chunk.page_content for chunk in chunks]
            embeddings = self.generate_embeddings(texts)
            
            # Store in ChromaDB
            self.store_in_chromadb(chunks, embeddings)
            
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting {pdf_path.name}: {str(e)}")
            return False
    
    def ingest_all_papers(self):
        """Ingest all PDFs in the research_papers directory."""
        if not RESEARCH_PAPERS_DIR.exists():
            logger.error(f"Research papers directory not found: {RESEARCH_PAPERS_DIR}")
            return
        
        pdf_files = list(RESEARCH_PAPERS_DIR.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("No PDF files found in research_papers directory")
            return
        
        logger.info("=" * 60)
        logger.info("Aether-Agent: Starting Knowledge Ingestion")
        logger.info("=" * 60)
        logger.info(f"Found {len(pdf_files)} PDF files to process\n")
        
        successful = 0
        failed = 0
        
        for pdf_path in pdf_files:
            logger.info(f"\nProcessing: {pdf_path.name}")
            if self.ingest_paper(pdf_path):
                successful += 1
            else:
                failed += 1
        
        # Get collection stats
        count = self.collection.count()
        
        logger.info("\n" + "=" * 60)
        logger.info("Ingestion Complete!")
        logger.info(f"  - Papers processed: {successful}")
        logger.info(f"  - Failed: {failed}")
        logger.info(f"  - Total chunks in database: {count}")
        logger.info("=" * 60)


def main():
    """Main ingestion function."""
    ingester = KnowledgeIngester()
    ingester.ingest_all_papers()


if __name__ == "__main__":
    main()
