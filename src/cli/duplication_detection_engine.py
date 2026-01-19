"""Main orchestration engine for code duplication detection pipeline.

This module coordinates the entire duplication detection workflow including
repository management, embedding generation, and similarity search.
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any

from .repository_manager import RepositoryManager
from .exporter import Exporter

from .embedding_engine import EmbeddingEngine
from .similarity_engine import SimilarityEngine
from config.settings import index_dir

logger = logging.getLogger(__name__)



class DuplicationDetectionEngine:
    """Main engine that orchestrates the code duplication detection pipeline.
    
    This class coordinates all phases of duplicate detection including:
    - Repository preparation and cloning
    - Code embedding and indexing
    - Similarity search and matching
    - LLM re-ranking for improved accuracy
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the duplication detection engine with configuration parameters.
        
        Args:
            **kwargs: Configuration parameters including:
                - candidate_repo: List of directories to analyze
                - rerank: Use LLM re-ranking for better accuracy
                - reasoning: Include LLM reasoning in results
                - similarity_threshold: Cosine similarity threshold for matches
          
        """
        # Configuration parameters
        self.candidate_repo: str = kwargs.get("candidate_repo")
        self.changed_files: str = kwargs.get("changed_files")
        self.rerank: bool = kwargs.get("rerank", False)
        self.reasoning: bool = kwargs.get("reasoning", False)
        self.similarity_threshold: float = kwargs.get("similarity_threshold", 0.85)


        # Initialize core components
        self.repo_manager = RepositoryManager()
        self.exporter = Exporter()
        self.embedding_engine = EmbeddingEngine()
        self.similarity_engine = SimilarityEngine(
            similarity_threshold=self.similarity_threshold,
            rerank=self.rerank,
            reasoning=self.reasoning
        )



    def run_pipeline(self) -> None:
        """Execute the complete code duplication detection pipeline.
        
        This method orchestrates the entire workflow:
        1. Repository preparation
        2. Code embedding and index generation
        3. Similarity search and duplicate detection
        """
        async def run_all() -> None:
            logger.info("Starting code duplication detection...")
            
            # Step 1: Prepare and validate code locations
            all_code_locations = self.repo_manager._prepare_code_locations(self.candidate_repo, self.changed_files)
            if all_code_locations is None:
                return  # Error already displayed by _prepare_code_locations
                
            try:
                # Step 3: Initialize and build indexes
                logger.info("Initializing indexes...")

                # Generate embeddings and build FAISS indexes
                candidate_repo = all_code_locations.get('candidate_repo')
                candidate_files = self.repo_manager.extract_candidate_files(candidate_repo)
                logger.debug(f"Candidate files: {candidate_files}")
                await self.embedding_engine.embed_candidate_corpus(candidate_files, None)
                        
                # Step 5: Get and process results
                
                changed_files = self.repo_manager.get_changed_files(all_code_locations.get('changed_files'))
                logger.debug(f"Changed files: {changed_files}")
                candidate_index = self.embedding_engine.get_candidate_index()
                logger.debug(f"Candidate index loaded")
               
                async for query_embeddings, chunks in self.embedding_engine.stream_query_embeddings(changed_files):
                    logger.debug(f"Query embeddings: {len(query_embeddings)} dimensions")
                    logger.debug(f"Chunks: {len(chunks)} blocks")

                    if chunks and isinstance(chunks, list):
                        search_results = await self.similarity_engine.run_semantic_search(query_embeddings, chunks, candidate_index)
                        logger.debug(f"Search results: {len(search_results)} matches")
                        # self.exporter.stream_diagonistics(search_results)
                        self.exporter.export_diagonistics(search_results)
                    
                completion_msg = "Code duplication detection complete!"
                logger.info(completion_msg)

            except Exception as e:
                logger.error(f"Failed to index: {e}", exc_info=True)

            finally:
                # self.exporter.finalize_json_export()
                self.repo_manager._clean_up('chunks.jsonl')
                
        # Execute the async pipeline
        asyncio.run(run_all())
    