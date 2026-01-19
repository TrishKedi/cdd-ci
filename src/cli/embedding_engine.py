"""Code embedding and FAISS index generation engine.

This module handles the extraction of code blocks from repositories,
generation of embeddings using AI services, and creation of FAISS
indexes for efficient similarity search.
"""

import os
import faiss
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from core.utils.helpers import get_index_path
from config.settings import index_dir
from core.services import CodeBaseProcessor, EmbeddingIndex, OpenAIService

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Manages code embedding generation and FAISS index creation.
    
    This class orchestrates the process of extracting code blocks from
    repositories, generating embeddings using AI services, and building
    FAISS indexes for efficient similarity search operations.
    """

    def __init__(self) -> None:
        """Initialize the embedding engine with required services.
        
        Sets up code processing, AI embedding service, and console output.
        """
        self.code_base_processor = CodeBaseProcessor()
        self.embedder = EmbeddingIndex()
        self.openai = OpenAIService()


    def get_candidate_index(self) -> faiss.Index:

        return self.embedder.get_index()

    async def get_current_index(self, code_directory: str) -> Optional[Dict[str, Any]]:
        """Retrieve existing index information for a repository.
        
        Args:
            code_directory: Path to the code repository
            
        Returns:
            Dictionary containing index metadata if exists, None otherwise
        """
        index_registry = await get_index(code_directory)
        return index_registry

    async def embed_candidate_corpus(
        self, 
        candidate_files: str, 
        status: Optional[Any]
    ) -> Dict[str, str]:
        """Process all repositories and create FAISS indexes.
        
        Iterates through all code repositories, checks for existing indexes,
        and builds new ones as needed. Returns mapping of index IDs to paths.
        
        Args:
            code_directory: List of repository paths to process
            status: Rich status object for progress updates
            
        Returns:
            Dictionary mapping index registry IDs to FAISS file paths
        """
      
        # Build new index for this repository
        logger.info("Building candidate index...")

        # Generate embeddings and build index
       
        await self.embed_code_blocks_in_batches(candidate_files, is_query=False)
        
        logger.info("Index built for candidate repo")

        # Display completion summary
        logger.info("Embedding complete!")
        

    async def embed_code_blocks_in_batches(
        self, 
        files: List[Path],
        is_query: bool
    ) -> None:

        """Process code blocks in batches to generate embeddings and build FAISS index.
        
        Implements a streaming pipeline where each batch is fully processed
        (extracted, embedded, and indexed) before moving to the next batch.
        This approach manages memory usage efficiently for large codebases.
        
        Args:
            code_directory: Path to the directory containing code files
            
        Returns:
            Tuple of (index_registry_id, index_path) for the processed repository
        """
        # Configuration
        batch_size = 50  # Adjust based on memory constraints
        
        # Initialize tracking variables
        total_blocks = 0

        batch = []
       
        # Set up index infrastructure
        # index_registry_id, faiss_file = await ingest_index(code_directory)
        # index_path = os.path.join(index_dir, faiss_file)
        
        
        # Process code blocks with progress tracking
        logger.info("Extracting code blocks from candidate repo...")
        
        # Extract and process code blocks in batches
        for code_chunk in self.code_base_processor.extract_code_chunks(files, is_candidate = (not is_query), batch_size=batch_size):
            batch.append(code_chunk)
            
            total_blocks += 1
            
            # Update progress periodically
            if total_blocks % 10 == 0:
                logger.debug(f"Extracted {total_blocks} code blocks from candidate repo...")
            
            if len(batch) >= batch_size:
             
                #Process complete batches
                embeddings = await self._process_batch(
                    batch, 
                    total_blocks,
                    is_query=is_query
                )

                batch = []


        if batch:
            #Process remaining batch
            embeddings = await self._process_batch(
                batch, 
                total_blocks,
                is_query=is_query
            )

      
        logger.info(f'Completed indexing {total_blocks} code blocks for candidate repo')
        
    async def stream_query_embeddings(
        self, 
        files: List[Path]
    ) -> None:

        """Process code blocks in batches to generate embeddings and build FAISS index.
        
        Implements a streaming pipeline where each batch is fully processed
        (extracted, embedded, and indexed) before moving to the next batch.
        This approach manages memory usage efficiently for large codebases.
        
        Args:
            code_directory: Path to the directory containing code files
            
        Returns:
            Tuple of (index_registry_id, index_path) for the processed repository
        """
        # Configuration
        batch_size = 50  # Adjust based on memory constraints
        
        # Initialize tracking variables
        total_blocks = 0

        batch = []
           
        # Process code blocks with progress tracking
        logger.info("Extracting code blocks for query...")
        
        # Extract and process code blocks in batches
        for code_chunk in self.code_base_processor.extract_code_chunks(files, is_candidate = False, batch_size=batch_size):
            batch.append(code_chunk)
            
            total_blocks += 1
            
            # Update progress periodically
            if total_blocks % 10 == 0:
                logger.debug(f"Extracted {total_blocks} code blocks...")
            
            if len(batch) >= batch_size:
             
                #Process complete batches
                embeddings = await self._process_batch(
                    batch, 
                    total_blocks,
                    is_query=True
                )

                batch = []
                yield embeddings, batch

        if batch:
            #Process remaining batch
            embeddings = await self._process_batch(
                batch, 
                total_blocks,
                is_query=True
            )

            yield embeddings, batch
                   
    async def _process_batch(
        self,
        code_blocks: List[Any],
        total_blocks: int,
        is_query: bool
        
    ) -> Optional[np.array]:
        """Process a batch of code blocks through the embedding pipeline.
        
        Args:
            code_blocks: List of code block objects to process
            code_directory: Source directory path
            index_registry_id: Database index identifier
            status: Rich status object for updates
            total_blocks: Current total count of processed blocks
        """
        
        # Generate embeddings for current batch
        logger.debug("Generating embeddings...")

      
        blocks = [code_block['processedCode'] for code_block in code_blocks]
        batch_embeddings = self.openai.generate_embeddings(blocks)

        if is_query:
            return batch_embeddings
        
        # Store embeddings in FAISS index
        logger.debug("Storing embeddings to FAISS index...")
        self.embedder.add_embeddings(batch_embeddings)
