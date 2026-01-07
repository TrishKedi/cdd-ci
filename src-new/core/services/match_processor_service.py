"""Match processing service for code similarity results.

This module handles processing and filtering of similarity search results,
including deduplication, threshold filtering, and match data enrichment.
"""
import json
from typing import List, Tuple, Dict, Any, Set, Optional
from core.database.ingest import get_embeddings


class MatchProcessor:
    """Processes and filters similarity search results.
    
    This class handles the post-processing of FAISS similarity search results,
    including filtering by thresholds, deduplication, and enrichment with
    code metadata from the database.
    """

    def filter_match_blocks(
        self, 
        blocks: List[Tuple], 
        faiss_vector_id: int
    ) -> Optional[Tuple]:
        """Filter code blocks by FAISS vector ID.
        
        Args:
            blocks: List of code block tuples from database
            faiss_vector_id: FAISS vector ID to match against
            
        Returns:
            Matching code block tuple, or None if not found
        """
        def filter_by_fvid(query_blocks: Tuple) -> bool:
            """Check if block matches the target FAISS vector ID."""
            return query_blocks[1] == faiss_vector_id

        filtered_blocks = list(filter(filter_by_fvid, blocks))
        
        if filtered_blocks:
            return filtered_blocks[0]
        
        return None

    def is_unique_match(
        self, 
        unique_matches: Set[Tuple[int, ...]], 
        neighbour_candidate: List[int]
    ) -> bool:
        """Check if a match pair is unique and add it to the set.
        
        Args:
            unique_matches: Set of previously seen match pairs
            neighbour_candidate: List containing query and candidate IDs
            
        Returns:
            True if this is a new unique match, False if already seen
        """
        # Create a normalized tuple for consistent comparison
        sorted_neighbour_candidate = tuple(sorted(neighbour_candidate))
        
        # Check if we've seen this match pair before
        if sorted_neighbour_candidate in unique_matches:
            return False

        # Add to set and return True for new unique match
        unique_matches.add(sorted_neighbour_candidate)
        return True

    async def process_matches(
        self,
        chunks: List[Dict[str, Any]], 
        search_results: Tuple[List[int], List[float]], 
        similarity_threshold: float
  
    ) -> Optional[Dict[str, Any]]:
        """Process similarity search results into structured match data.
        
        Args:
            query_pstn: Query position in the index
            query_index_id: Query index identifier
            cand_index_id: Candidate index identifier
            result: Tuple of (neighbor_ids, similarity_scores) from FAISS
            unique_matches: Set to track unique match pairs
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            Dictionary containing processed match data, or None if no valid query block
        """
        # Fetch code blocks from database
       
        # cand_blocks = await get_embeddings(cand_index_id)
        
        # Extract similarity results
        neighbours, scores = search_results
        neighbour_similarity_map = list(zip(neighbours, scores))

        
        # Get the query code block
        query_block = self.filter_match_blocks(chunks, query_pstn)
        if not query_block:
            return None

        # Extract query metadata
        query_path = query_block[2]
        query_code = query_block[3]
        query_start = query_block[4]
        
        # Process candidate matches
        processed_matches = self._build_candidate_matches(
            neighbour_similarity_map,
            query_pstn,
            similarity_threshold
        )

        return {
            'code': query_code,
            'path': query_path,
            'start': query_start,
            'matches': processed_matches
        }
    
    def _build_candidate_matches(
        self,
        neighbour_similarity_map: List[Tuple[int, float]],
        cand_blocks: List[Tuple],
        unique_matches: Set[Tuple[int, ...]],
        query_pstn: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Build list of candidate matches with metadata.
        
        Args:
            neighbour_similarity_map: List of (neighbor_id, score) pairs
            cand_blocks: Candidate code blocks from database
            unique_matches: Set to track unique matches
            query_pstn: Query position for uniqueness checking
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of processed candidate match dictionaries
        """
        matches = []
        
        for neighbour_id, score in enumerate(neighbour_similarity_map):
            # Apply filters: threshold, uniqueness, and valid candidate block
            if (score >= similarity_threshold and 
                self.is_unique_match(unique_matches, [query_pstn, neighbour_id])):
                
                candidate_block = self.filter_match_blocks(cand_blocks, neighbour_id)
                if candidate_block:
                    matches.append({
                        "path": candidate_block[2],
                        "score": score,
                        "code": candidate_block[3],
                        "start": candidate_block[4],
                    })
        
        return matches

    def replace_candidates(self, search_results):
        print(search_results[0])
        print(search_results[0][0])
        # print(search_results[0][0].index(30))

        new_cand = []

        def transform_candidates(line, search_result):
            c_id = line.get('id')  
            candidates = search_result[0]
            scores = search_result[1]
            if c_id in candidates:                  
                c_pstn = candidates.index(c_id)
                score = scores[c_pstn]
                candidates[c_pstn] = {
                    **line,
                    "score": score
                }
            return candidates

        with open('chunks.jsonl', 'r') as f:
            for line in f:
                cand_dets = json.loads(line)
                # print("=====LINE=====")
                # print(line)
                
                new_cand = list(map(lambda cands: transform_candidates(cand_dets, cands), search_results))
                # print(new_cand)
                # print(len(new_cand))

        return new_cand

    def replace_queries(self, chunks, candidates):
    
        return [
            {
                **chunk,
                "matches": candidates[i]
            }
            for i, chunk in enumerate(chunks)
        ]
       
       

