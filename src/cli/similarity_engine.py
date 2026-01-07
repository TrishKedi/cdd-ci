"""Similarity search engine for code duplication detection.

This module orchestrates the similarity search process including
FAISS index lookups, match processing, LLM re-ranking, and result
export/visualization.
"""

import logging
import numpy as np
import faiss
from typing import Dict, Set, Any, List, Optional, AsyncGenerator, Tuple
from rich.console import Console

from .exporter import Exporter
from .visualizer import Visualizer
from config.settings import default_similarity_threshold
from core.services import SimilarityLookup, MatchProcessor


# Set up logging
logger = logging.getLogger(__name__)


class SimilarityEngine:
    """Orchestrates similarity search across multiple FAISS indexes for code duplication detection.
    
    This class manages the complete similarity search workflow, providing both fast streaming
    mode for immediate results and batched processing with LLM re-ranking for improved accuracy.
    It serves as the main orchestrator for the code duplication detection pipeline.
    
    Key Features:
        - FAISS-based vector similarity search across multiple code repositories
        - Intelligent match processing and deduplication to avoid false positives
        - Optional LLM re-ranking using Claude for semantic similarity validation
        - Real-time result streaming to web interface for interactive exploration
        - Configurable export formats (JSON, JSONL) with automatic file management
        - Rich terminal visualization with progress tracking and match display
    
    Processing Modes:
        - Streaming Mode: Fast, immediate results without LLM validation (default)
        - Batched Mode: LLM re-ranking for higher accuracy but slower processing
    
    Attributes:
        index_path_id_map (Dict[str, str]): Mapping of index IDs to FAISS file paths
        export (bool): Whether to export results to files
        rerank (bool): Whether to use LLM re-ranking for improved accuracy
        serve (bool): Whether to stream results to web interface
        verbose (bool): Whether to show detailed output and debugging info
        reasoning (bool): Whether to include LLM reasoning in results
        less (bool): Whether to use less pager for terminal output
        similarity_threshold (float): Minimum similarity score for matches
        similarity_lookup (SimilarityLookup): Core FAISS search service
        visualizer (Visualizer): Terminal output formatting service
        llm_reranker (LLMReranker): LLM-based semantic similarity validator
        exporter (Exporter): Result export and file management service
        match_processor (MatchProcessor): Match filtering and deduplication service
        console (Console): Rich console for formatted terminal output
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the similarity engine with configuration options.
        
        Sets up all necessary service components and configuration parameters for
        the similarity search workflow. The engine can operate in different modes
        depending on the provided configuration.
        
        Args:
            index_path_id_map (Dict[str, str]): Dictionary mapping repository/index IDs 
                to their corresponding FAISS index file paths. Each key represents a 
                unique identifier for a code repository, and each value is the absolute 
                path to the FAISS index file containing vectorized code embeddings.
            **kwargs (Any): Configuration options for customizing behavior:
                - export (bool, optional): Enable automatic export of results to JSON/JSONL 
                    files. Defaults to False.
                - rerank (bool, optional): Enable LLM-based semantic re-ranking of matches 
                    for improved accuracy. Defaults to False.
                - serve (bool, optional): Enable real-time streaming of results to web 
                    interface for interactive exploration. Defaults to False.
                - verbose (bool, optional): Show detailed progress information and debugging 
                    output in terminal. Defaults to False.
                - reasoning (bool, optional): Include LLM reasoning explanations in match 
                    results when re-ranking is enabled. Defaults to False.
                - similarity_threshold (float, optional): Minimum cosine similarity score 
                    (0.0-1.0) for considering matches. Defaults to default_similarity_threshold.
                - less (bool, optional): Use less pager for paginated terminal output of 
                    large result sets. Defaults to False.
        
        Note:
            The similarity_threshold parameter significantly affects both performance and 
            accuracy. Lower values increase recall but may introduce more false positives.
        """
        # Core configuration - index mapping for FAISS similarity search
        self.index_path_id_map: Optional[Dict[str, str]] = None
        
        # Feature flags controlling processing behavior
        self.export: bool = kwargs.get("export", False)  # Enable file export
        self.rerank: bool = kwargs.get("rerank", False)  # Enable LLM re-ranking
        self.serve: bool = kwargs.get("serve", False)    # Enable web streaming
        
        # Output and display configuration
        self.verbose: bool = kwargs.get("verbose", False)     # Detailed logging
        self.reasoning: bool = kwargs.get("reasoning", False) # Include LLM explanations
        self.less: bool = kwargs.get("less", False)           # Use pager for output
        
        # Similarity matching threshold (affects precision vs recall trade-off)
        self.similarity_threshold: float = kwargs.get(
            "similarity_threshold", 
            default_similarity_threshold
        )

        # Initialize core service components with dependency injection pattern
        self.similarity_lookup: SimilarityLookup = SimilarityLookup(self.similarity_threshold)
        self.visualizer: Visualizer = Visualizer(
            verbose=self.verbose, 
            reasoning=self.reasoning, 
            use_pager=self.less
        )

        self.exporter: Exporter = Exporter(auto_open=True, verbose=True)
        self.match_processor: MatchProcessor = MatchProcessor()
        self.console: Console = Console()

    async def run_semantic_search(self, query: np.array, chunks: List[Dict[str, str]], index: faiss.Index) -> List[Tuple[str, str]]:
        search_results  = self.similarity_lookup.run_semantic_search(query, index)
        candidates = self.match_processor.replace_candidates(search_results)
        # print(candidates[0])
        print(chunks)
        print(len(chunks))
        print(chunks[0])
        return [
            {
                **chunk,
                "matches": candidates[i]
            }
            for i, chunk in enumerate(chunks)
        ]
        

        # matches = await self.match_processor.process_matches(
        #             chunks, 
        #             search_results, 
        #             self.similarity_threshold
        #         )
        # return matches

    async def run_exhaustive_similarity_lookup(self) -> None:
        """Execute the complete similarity search workflow across all configured indexes.
        
        This is the main entry point for similarity search operations. It orchestrates
        the entire pipeline from FAISS index searching through result processing and
        export. The method automatically selects between streaming and batched processing
        modes based on the re-ranking configuration.
        
        Processing Flow:
            1. Display configuration and initialize export files if needed
            2. Choose processing mode (streaming vs batched with LLM re-ranking)
            3. Execute similarity search across all index pairs
            4. Process, filter, and deduplicate matches
            5. Optional LLM re-ranking for semantic validation
            6. Display results and export to configured formats
            7. Stream results to web interface if enabled
        
        Note:
            - Streaming mode: Fast results, no LLM validation, immediate display
            - Batched mode: Slower but more accurate with LLM semantic validation
            - Export files are automatically managed (initialized/finalized)
            - Progress is tracked and displayed in real-time
        
        Raises:
            Exception: If FAISS index loading fails or LLM services are unavailable
                when re-ranking is enabled
        """
        # Initialize deduplication tracking across all index pairs
        unique_matches: Set[str] = set()
        
        # Log workflow initiation
        self.console.print("Started similarity lookup...", style="bold blue")
        logger.info("Starting exhaustive similarity lookup workflow")
        
        # Display configuration information for user awareness
        if self.similarity_threshold != default_similarity_threshold:
            self.console.print(f"Using custom similarity threshold: {self.similarity_threshold}", style="yellow")
        # Initialize export infrastructure if file output is enabled
        if self.export:
            self.console.print("Initializing export files...", style="dim")
            try:
                self.exporter.initialize_export()
                logger.info("Export files initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize export files: {e}")
                self.console.print(f" Export initialization failed: {e}", style="red")
                # Continue without export rather than failing completely
                self.export = False

        try:
            # Choose processing mode based on accuracy vs speed requirements
            if self.rerank:
                # Use batched processing for LLM re-ranking (slower but more accurate)
                self.console.print(" Using LLM re-ranking...", style="yellow")
                logger.info("Starting batched processing with LLM re-ranking")
                await self.run_exhaustive_similarity_lookup_batched(unique_matches, batch_size=10)
            else:
                # Use fast streaming mode for immediate results (faster but less accurate)
                self.console.print(
                    "Using fast streaming mode (no LLM re-ranking)...", 
                    style="green"
                )
                logger.info("Starting streaming processing without LLM re-ranking")
                await self.run_exhaustive_similarity_lookup_streaming(unique_matches)
        except Exception as e:
            logger.error(f"Similarity lookup workflow failed: {e}", exc_info=True)
            raise
        finally:
            # Ensure export cleanup happens even if processing fails
            if self.export:
                self.console.print("Finalizing export files...", style="dim")
                self.exporter.finalize_export()
                logger.info("Export files finalized")


    async def run_exhaustive_similarity_lookup_streaming(self, unique_matches: Set[str]) -> None:
        """Execute streaming similarity search without LLM re-ranking for fast results.
        
        This method implements the fast path for similarity search, processing matches
        as they are found and displaying results immediately. It's optimized for speed
        and real-time feedback but doesn't include semantic validation via LLM.
        
        Args:
            unique_matches (Set[str]): Set for tracking unique matches to prevent 
                duplicates across different index pairs. Modified in-place during processing.
        
        Processing Flow:
            1. Iterate through all index pairs using async generator
            2. Process each match through MatchProcessor for filtering
            3. Display matches immediately when found (if not serving to web)
            4. Export matches to files if export is enabled
            5. Stream matches to web interface if serve mode is enabled
            6. Update progress indicators every 10 processed items
        
        Performance Characteristics:
            - Low latency: Results displayed as soon as found
            - Memory efficient: No batching, immediate processing
            - High throughput: No LLM API calls blocking the pipeline
            - Real-time progress: Continuous status updates
        
        Note:
            This method prioritizes speed over accuracy. For semantic validation
            of matches, use the batched processing mode with LLM re-ranking.
        """
        # Initialize counters for progress tracking
        total_processed: int = 0
        matches_found: int = 0
        
        # Use rich console status for real-time progress display
        with self.console.status("[bold green]Searching for duplicate code patterns...", spinner="dots") as status:
            # Async iteration over all possible index pairs for similarity search
            async for query_pstn, query_index_id, cand_index_id, result in self.similarity_lookup.generate_lookup_results(self.index_path_id_map):
                # Process raw FAISS results through match processor for filtering and deduplication
                matches = await self.match_processor.process_matches(
                    query_pstn, query_index_id, cand_index_id, result, 
                    unique_matches, self.similarity_threshold
                )
                total_processed += 1
                
                # Update progress display every 10 items to avoid excessive UI updates
                if total_processed % 10 == 0:
                    status.update(f"[bold green]Processed {total_processed} code blocks, found {matches_found} matches...")
                    # Notify web interface of progress updates for real-time dashboard
                    if self.serve:
                        await stream_progress_to_web(total_processed, matches_found)

                # Process matches if any were found after filtering
                if matches and matches.get("matches"):
                    matches_found += len(matches["matches"])

                    # Temporarily pause progress indicator to display match details cleanly
                    status.stop()
                    # Display matches in terminal only if not serving to web (avoid duplicate output)
                    if not self.serve:
                        self.visualizer.display_matches(matches)
                    status.start()

                    # Export matches to configured file formats
                    if self.export:
                        self.exporter.export(matches)
                        
        # Display completion summary with final statistics
        self.console.print(
            f"✅ Completed similarity search: {total_processed} blocks processed, {matches_found} matches found", 
            style="green"
        )
        logger.info(f"Streaming similarity search completed: {total_processed} blocks, {matches_found} matches")


    async def run_exhaustive_similarity_lookup_batched(self, unique_matches: Set[str], batch_size: int = 10) -> None:
        """Execute batched similarity search with LLM re-ranking for improved accuracy.
        
        This method implements the high-accuracy path for similarity search, collecting
        matches into batches before sending them through LLM-based semantic validation.
        While slower than streaming mode, it provides significantly better precision
        by filtering out false positives through semantic analysis.
        
        Args:
            unique_matches (Set[str]): Set for tracking unique matches to prevent 
                duplicates across different index pairs. Modified in-place during processing.
            batch_size (int, optional): Number of matches to collect before sending 
                to LLM for re-ranking. Defaults to 10. Larger batches are more efficient 
                but use more memory and have higher latency.
        
        Processing Flow:
            1. Iterate through all index pairs collecting matches
            2. Buffer matches until batch_size is reached
            3. Send batch to LLM re-ranker for semantic validation
            4. Process and display re-ranked results
            5. Export validated matches if enabled
            6. Handle remaining matches in final partial batch
        
        Performance Characteristics:
            - Higher accuracy: LLM semantic validation reduces false positives
            - Batch efficiency: Multiple matches processed in single LLM call
            - Memory buffering: Temporary storage of matches before processing
            - Adaptive batching: Handles partial final batches gracefully
        
        Note:
            Batch size affects the trade-off between API efficiency and memory usage.
            Larger batches reduce API calls but increase memory consumption and latency.
        """
        # Initialize batch processing state
        batch_buffer: List[Dict[str, Any]] = []  # Buffer for accumulating matches
        total_processed: int = 0                 # Total code blocks processed
        batches_processed: int = 0               # Number of LLM batches completed
        total_matches: int = 0                   # Total matches found across all batches
        
        # Use rich console status with distinct styling for batched mode
        with self.console.status("[bold yellow]Searching and preparing batches for LLM re-ranking...", spinner="dots") as status:
            # Async iteration over all possible index pairs for similarity search
            async for query_pstn, query_index_id, cand_index_id, result in self.similarity_lookup.generate_lookup_results(self.index_path_id_map):
                # Process raw FAISS results through match processor
                matches = await self.match_processor.process_matches(
                    query_pstn, query_index_id, cand_index_id, result, 
                    unique_matches, self.similarity_threshold
                )
                total_processed += 1
                
                # Update progress less frequently (every 20 items) since batching has higher overhead
                if total_processed % 20 == 0:
                    status.update(f"[bold yellow]Processed {total_processed} blocks, prepared {batches_processed} batches...")
                    # Notify web interface of progress for dashboard updates
                    if self.serve:
                        await stream_progress_to_web(total_processed, total_matches)
                
                # Only buffer matches that contain actual candidates (skip empty results)
                if matches and matches.get("matches"):
                    batch_buffer.append(matches)
                    total_matches += len(matches["matches"])
                
                # Process accumulated batch when it reaches target size
                if len(batch_buffer) >= batch_size:
                    status.stop()
                    # Process batch through LLM re-ranking pipeline
                    await self.rerank_batch(batch_buffer)
                    batches_processed += 1
                    batch_buffer = []  # Clear buffer for next batch accumulation
                    status.start()
            
            # Handle remaining matches in partial final batch
            if batch_buffer:
                status.stop()
                await self.rerank_batch(batch_buffer)
                batches_processed += 1
                
        # Display completion summary with batch processing statistics
        self.console.print(
            f"✅ Completed similarity search: {total_processed} blocks processed, {total_matches} total matches with LLM re-ranking", 
            style="green"
        )
        logger.info(f"Batched similarity search completed: {total_processed} blocks, {batches_processed} batches, {total_matches} matches")

    async def rerank_batch(self, matches_batch: List[Dict[str, Any]]) -> None:
        """Process a batch of matches through LLM re-ranking and handle results.
        
        Takes a collected batch of similarity matches and processes them through
        the LLM re-ranker for semantic validation. This helps filter out false
        positives by using Claude's understanding of code semantics and intent.
        
        Args:
            matches_batch (List[Dict[str, Any]]): List of match dictionaries to be 
                re-ranked. Each dictionary contains match metadata, code snippets, 
                and similarity scores from the initial FAISS search.
        
        Processing Flow:
            1. Send entire batch to LLM re-ranker service
            2. LLM evaluates semantic similarity of each code pair
            3. Matches are filtered/scored based on LLM assessment
            4. Display validated matches in terminal (if not serving to web)
            5. Export validated matches to files if enabled
            6. Stream validated matches to web interface if enabled
        
        Note:
            The LLM re-ranker uses multi-threading internally to process multiple
            code pairs concurrently, improving efficiency over sequential processing.
            Results maintain original match structure but with updated confidence scores.
        """
        
        # Display progress for LLM processing with distinct visual styling
        # with self.console.status("[bold magenta]LLM re-ranking matches...", spinner="dots"):
        #     # Send entire batch to LLM service for semantic validation
        #     # This uses multi-threading internally to process code pairs concurrently
        #     reranked_matches: List[Dict[str, Any]] = await self.llm_reranker.rerank_matches_batch(matches_batch)
        
        # # Process each re-ranked match in the batch
        # for matches in reranked_matches:
        #     # Display matches in terminal only if not serving to web interface
        #     # This avoids duplicate output when web interface is active
        #     if not self.serve:
        #         self.visualizer.display_matches(matches)

        #     # Export validated matches to configured file formats
        #     if self.export:
        #         self.exporter.export(matches)

