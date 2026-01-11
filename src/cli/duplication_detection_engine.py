"""Main orchestration engine for code duplication detection pipeline.

This module coordinates the entire duplication detection workflow including
repository management, embedding generation, and similarity search.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.status import Status

from .repository_manager import RepositoryManager
from .exporter import Exporter

from .embedding_engine import EmbeddingEngine
from .similarity_engine import SimilarityEngine
from config.settings import index_dir



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
        self.console = Console()



    def run_pipeline(self) -> None:
        """Execute the complete code duplication detection pipeline.
        
        This method orchestrates the entire workflow:
        1. Repository preparation
        2. Code embedding and index generation
        3. Similarity search and duplicate detection
        """
        async def run_all() -> None:
            # self.console.print("Starting code duplication detection...", style="bold blue")
            
            # Step 1: Prepare and validate code locations
            all_code_locations = self.repo_manager._prepare_code_locations(self.candidate_repo, self.changed_files)
            # print(all_code_locations)
            if all_code_locations is None:
                return  # Error already displayed by _prepare_code_locations
                
            try:
                # Step 3: Initialize and build indexes
                with self.console.status("[bold green]Initializing indexes...", spinner="dots") as status:
                    # Rebuild indexes if requested
                
                    status.update("[bold green]Loading repositories...")

                    # Generate embeddings and build FAISS indexes
                    candidate_repo = all_code_locations.get('candidate_repo')
                    candidate_files = self.repo_manager.extract_candidate_files(candidate_repo)
                    await self.embedding_engine.embed_candidate_corpus(candidate_files, status)
                        
                # # Step 4: Run similarity search (unless embed-only mode)
                # if not self.embed_only:
                #     self.console.print(f"Started exhaustive similarity search across {len(index_path_id_map)} repositories...", style="bold blue")
                #     await self.start_semantic_search(index_path_id_map)
                
                # Step 5: Display completion status
                
                changed_files = self.repo_manager.get_changed_files(all_code_locations.get('changed_files'))
                # print(changed_files)
                candidate_index = self.embedding_engine.get_candidate_index()
                # print(candidate_index)
               
                async for query_embeddings, chunks in self.embedding_engine.stream_query_embeddings(changed_files):
                    # print(f"\n{query_embeddings}\n")
                    # print(f"\n{chunks}\n")

                    if chunks and isinstance(chunks, list):
                        search_results = await self.similarity_engine.run_semantic_search(query_embeddings, chunks, candidate_index)
                        # print(search_results)
                        self.exporter.stream_diagonistics(search_results)
                        # self.exporter.export_rdjson(search_results)
                    
                completion_msg = "Code duplication detection complete!"
                # self.console.print(f"\n{completion_msg}", style="bold green")

            except Exception as e:
                pass
                # self.console.print(f"Failed to index: {e}")

            finally:
                # self.exporter.finalize_json_export()
                self.repo_manager._clean_up('chunks.jsonl')
                
        # Execute the async pipeline
        asyncio.run(run_all())
    