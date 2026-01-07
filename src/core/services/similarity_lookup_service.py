"""FAISS-based similarity search for code embeddings.

This module provides efficient similarity lookup capabilities using
FAISS indexes to find similar code blocks across repositories.
"""

import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, AsyncGenerator, Any

import faiss
import numpy as np


class SimilarityLookup:
    """Handles similarity search operations using FAISS indexes.
    
    This class provides methods to search for similar code embeddings
    across multiple FAISS indexes efficiently using parallel processing.
    """

    def __init__(self, similarity_threshold: float) -> None:
        """Initialize similarity lookup with threshold configuration.
        
        Args:
            similarity_threshold: Minimum similarity score for matches
        """
        self.similarity_threshold = similarity_threshold

    def run_semantic_search(
        self, 
        query_embeddings: np.ndarray, 
        faiss_index: faiss.Index
    ) -> List[Tuple[List[int], List[float]]]:
        """Perform semantic search using FAISS index.
        
        Args:
            query_embeddings: Query vectors for similarity search
            faiss_index: FAISS index to search against
            
        Returns:
            List of tuples containing (neighbor_ids, similarity_scores)
        """
        similarity_scores, neighbours = faiss_index.search(query_embeddings, k=1)
        search_results = list(zip(neighbours.tolist(), similarity_scores.tolist()))
        
        # print(type(similarity_scores), type(neighbours))
        # print(len(similarity_scores))
        # print(similarity_scores.tolist())

        
        return search_results

    def search_single_shard(
        self, 
        candidate_index_path: str, 
        query_index_path: str
    ) -> List[Tuple[List[int], List[float]]]:
        """Search a single FAISS index shard for similar embeddings.
        
        Args:
            candidate_index_path: Path to the candidate FAISS index
            query_index_path: Path to the query FAISS index
            
        Returns:
            Search results containing neighbor IDs and similarity scores
        """
        # Configure FAISS for single-threaded operation
        faiss.omp_set_num_threads(1)

        # Load FAISS indexes
        query_faiss_index = faiss.read_index(query_index_path)
        candidate_faiss_index = faiss.read_index(candidate_index_path)
        
        # Extract query embeddings
        n = query_faiss_index.ntotal
        q_emb = query_faiss_index.reconstruct_batch(np.arange(n, dtype=np.int64))
        query = np.ascontiguousarray(q_emb, dtype=np.float32)
        
        # Perform semantic search
        search_results = self.run_semantic_search(query, candidate_faiss_index)
        return search_results


    async def search_all_shards(
        self, 
        faiss_indexes: List[str], 
        query_index_path: str
    ) -> List[List[Tuple[List[int], List[float]]]]:
        """Search all FAISS index shards in parallel.
        
        Args:
            faiss_indexes: List of paths to candidate FAISS indexes
            query_index_path: Path to the query FAISS index
            
        Returns:
            List of search results from all shards
        """
        loop = asyncio.get_running_loop()
        
        # Optimize thread pool size based on available resources
        max_workers = min(len(faiss_indexes), os.cpu_count() or 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create parallel search tasks
            tasks = [
                loop.run_in_executor(
                    executor, 
                    self.search_single_shard,
                    cand_index_path, 
                    query_index_path
                )
                for cand_index_path in faiss_indexes
            ]
            
            # Execute all searches concurrently
            results = await asyncio.gather(*tasks)

        return results


    async def generate_lookup_results(
        self, 
        index_path_id_map: Dict[str, str]
    ) -> AsyncGenerator[Tuple[int, str, str, Tuple[List[int], List[float]]], None]:
        """Generate similarity lookup results across all index combinations.
        
        Performs exhaustive similarity search by using each index as both
        query and candidate, ensuring all possible matches are found.
        
        Args:
            index_path_id_map: Dictionary mapping index IDs to file paths
            
        Yields:
            Tuples of (query_position, query_index_id, candidate_index_id, search_result)
        """
        faiss_indexes = list(index_path_id_map.values())
        index_keys = list(index_path_id_map.keys())
    
        # Use each index as a query against all candidates
        for query_index_id, index_path in index_path_id_map.items():
            # Search this query index against all candidate indexes
            shard_results = await self.search_all_shards(faiss_indexes, index_path)
    
            # Process results from each candidate shard
            for cand_position, cand_result in enumerate(shard_results): 
                cand_index_id = index_keys[cand_position]
    
                # Yield each individual search result
                for query_position, result in enumerate(cand_result):
                    yield query_position, query_index_id, cand_index_id, result

            # Remove processed query from future candidate sets
            # This prevents duplicate comparisons (A->B and B->A)
            faiss_indexes.remove(index_path)
            index_keys.remove(query_index_id)
       






        





        

  


        


