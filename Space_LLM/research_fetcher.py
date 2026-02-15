"""
Aether-Agent: Automatic Paper Acquisition Module
Downloads aerospace research papers from arXiv and maintains a library manifest.
"""

import os
import json
import arxiv
from pathlib import Path
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Research keywords for aerospace papers
RESEARCH_KEYWORDS = [
    "Large Language Models for Aerospace Engineering",
    "Satellite Image Segmentation for Forest Fire Detection",
    "Hohmann Transfer Trajectory Optimization for CubeSats",
    "Small Satellite Thermal Management Systems"
]

# Directory setup - Use absolute path
RESEARCH_PAPERS_DIR = Path("research_papers").resolve()
MANIFEST_FILE = Path("library_manifest.json").resolve()
RESEARCH_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Research papers directory: {RESEARCH_PAPERS_DIR}")


def load_manifest() -> Dict[str, Dict]:
    """Load existing library manifest to track downloaded papers."""
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_manifest(manifest: Dict[str, Dict]):
    """Save library manifest to disk."""
    with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def download_paper(paper: arxiv.Result, manifest: Dict[str, Dict]) -> bool:
    """
    Download a paper if not already in manifest.
    Returns True if downloaded, False if already exists.
    """
    paper_id = paper.entry_id.split('/')[-1]
    
    # Check if already downloaded
    if paper_id in manifest:
        logger.info(f"Paper already in library: {paper.title[:60]}...")
        return False
    
    try:
        # Download PDF
        pdf_path = RESEARCH_PAPERS_DIR / f"{paper_id}.pdf"
        paper.download_pdf(dirpath=str(RESEARCH_PAPERS_DIR), filename=f"{paper_id}.pdf")
        
        # Add to manifest
        manifest[paper_id] = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "published": paper.published.strftime("%Y-%m-%d"),
            "year": paper.published.year,
            "summary": paper.summary[:200] + "..." if len(paper.summary) > 200 else paper.summary,
            "arxiv_id": paper_id,
            "pdf_path": str(pdf_path)
        }
        
        logger.info(f"✓ Downloaded: {paper.title[:60]}... ({paper.published.year})")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {paper.title}: {str(e)}")
        return False


def fetch_papers_for_keyword(keyword: str, max_results: int = 10) -> List[arxiv.Result]:
    """Search arXiv for papers matching the keyword."""
    logger.info(f"Searching arXiv for: '{keyword}'")
    
    import time
    
    # Add delay to avoid rate limiting
    time.sleep(5)  # Wait 5 seconds between searches
    
    search = arxiv.Search(
        query=keyword,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    try:
        results = list(search.results())
        logger.info(f"Found {len(results)} papers for '{keyword}'")
        return results
    except Exception as e:
        logger.warning(f"Rate limited or error for '{keyword}': {str(e)}")
        logger.info("Waiting 30 seconds before retry...")
        time.sleep(30)
        try:
            results = list(search.results())
            logger.info(f"Found {len(results)} papers for '{keyword}' (retry successful)")
            return results
        except:
            logger.error(f"Failed to fetch papers for '{keyword}' after retry")
            return []


def main():
    """Main function to fetch and download papers for all keywords."""
    logger.info("=" * 60)
    logger.info("Aether-Agent: Starting Automatic Paper Acquisition")
    logger.info("=" * 60)
    
    manifest = load_manifest()
    initial_count = len(manifest)
    
    all_papers = []
    
    # Fetch papers for each keyword
    import time
    for idx, keyword in enumerate(RESEARCH_KEYWORDS):
        if idx > 0:
            logger.info(f"Waiting 10 seconds before next search...")
            time.sleep(10)  # Wait between keywords
        papers = fetch_papers_for_keyword(keyword, max_results=10)
        all_papers.extend(papers)
    
    # Remove duplicates (same arxiv ID)
    seen_ids = set()
    unique_papers = []
    for paper in all_papers:
        paper_id = paper.entry_id.split('/')[-1]
        if paper_id not in seen_ids:
            seen_ids.add(paper_id)
            unique_papers.append(paper)
    
    logger.info(f"\nTotal unique papers found: {len(unique_papers)}")
    
    # Download papers
    downloaded_count = 0
    for paper in unique_papers:
        if download_paper(paper, manifest):
            downloaded_count += 1
    
    # Save manifest
    save_manifest(manifest)
    
    final_count = len(manifest)
    logger.info("\n" + "=" * 60)
    logger.info(f"Acquisition Complete!")
    logger.info(f"  - Papers in library: {final_count}")
    logger.info(f"  - New papers downloaded: {downloaded_count}")
    logger.info(f"  - Skipped (duplicates): {len(unique_papers) - downloaded_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
