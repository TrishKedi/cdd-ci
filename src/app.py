"""CLI application for code duplication detection.

This module provides the main entry point for the code duplication detector
command-line interface using Typer.
"""

import logging
import typer
from typing import List, Optional
from cli import DuplicationDetectionEngine
from config.settings import default_similarity_threshold

# Configure logging to output to terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = typer.Typer(
    name="find-duplicates",
    help="Advanced code duplication detection with AI-powered similarity analysis"
)


@app.command()
def find_duplicates(
    code_locations: Optional[List[str]] = typer.Argument(
        default=None, 
        help="Path to code directories to analyze for duplicates"
    ),
    verbose: bool = typer.Option(
        False, 
        help="Show detailed code snippets in output"
    ),
    rebuild: bool = typer.Option(
        False, 
        help="Force rebuild of existing indexes for given repositories"
    ),
    export: bool = typer.Option(
        False, 
        help="Export duplicate matches to JSON file"
    ),
    rerank: bool = typer.Option(
        False, 
        help="Use LLM re-ranking to filter false positive matches"
    ),
    reasoning: bool = typer.Option(
        False, 
        help="Include LLM reasoning explanations in re-ranking results"
    ),
    embed_only: bool = typer.Option(
        False, 
        "--embed-only", 
        help="Only build embeddings and indexes, skip similarity search"
    ),
    similarity_threshold: float = typer.Option(
        default_similarity_threshold, 
        "--similarity-threshold", 
        help=f"Cosine similarity threshold for matches (default: {default_similarity_threshold})"
    ),
    clone: Optional[List[str]] = typer.Option(
        None, 
        "--clone", 
        help="Git repository URLs to clone and process (can be used multiple times)"
    ),
    serve: bool = typer.Option(
        False, 
        "--serve", 
        help="Start interactive web server to display results in browser"
    ),
    port: int = typer.Option(
        8080, 
        "--port", 
        help="Web server port number (default: 8080)"
    ),
    auto_open: bool = typer.Option(
        True, 
        "--auto-open/--no-auto-open", 
        help="Automatically open web browser when serving (default: True)"
    ),
    less: bool = typer.Option(
        False,
        "--less",
        help="Use less pager to display output (prevents truncation)"
    ),
    changed_files: Optional[str] = typer.Option(
        None,
        "--changed-files",
        help="File with a list of changed files"
    ),
    candidate_repo: Optional[str] = typer.Option(
        None,
        "--candidate-repo",
        help="Repository that is to be searched for clones"
    )
) -> None:
    """Find duplicate code patterns across repositories using AI-powered similarity analysis.
    
    This command analyzes code repositories to identify duplicate or highly similar
    code blocks using advanced embedding techniques and optional LLM re-ranking.
    
    Examples:
        # Analyze local directories
        find-duplicates /path/to/repo1 /path/to/repo2
        
    """
    # Collect all command arguments for engine initialization
    args = locals()
    
    # Initialize and run the duplication detection pipeline
    duplication_detection_engine = DuplicationDetectionEngine(**args)
    duplication_detection_engine.run_pipeline()
    


if __name__ == "__main__":
    app()

