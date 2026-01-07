"""LLM-based re-ranking service for code similarity results.

This module provides intelligent re-ranking of code similarity matches using
Large Language Models to filter false positives and improve match accuracy.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from .sap_ai_core_service import SapAiCore


class LLMReranker:
    """Re-ranks code similarity matches using Large Language Model evaluation.
    
    This class uses LLM services to perform semantic analysis of code matches,
    filtering out false positives and providing confidence scores and reasoning
    for improved duplicate detection accuracy.
    """
    def __init__(
        self, 
        model: str = "claude-3-5-sonnet-20241022", 
        similarity_threshold: float = 0.85
    ) -> None:
        """Initialize the LLM re-ranker with configuration.
        
        Args:
            model: LLM model identifier for evaluation
            similarity_threshold: Minimum similarity score for re-ranking consideration
        """
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.max_workers = 3  # Limit concurrent API calls to prevent rate limiting
        self.sap_ai_core = SapAiCore()
        
    async def rerank_matches(self, matches: Dict[str, Any]) -> Dict[str, Any]:
        """Re-rank similarity matches using LLM evaluation to filter false positives.
        
        Args:
            matches: Dictionary containing code and similarity matches
            
        Returns:
            Updated matches dictionary with LLM-verified results
        """
        # Skip if no matches to process
        if not matches.get("matches"):
            return matches
            
        query_code = matches["code"]
        candidates = matches["matches"]

        # Consider all candidates for re-ranking
        high_score_candidates = list(candidates)
        
        if not high_score_candidates:
            return matches
            
        # Process candidates in parallel with controlled concurrency
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    self._evaluate_similarity_sync,
                    query_code,
                    candidate
                )
                for candidate in high_score_candidates
            ]
            
            # Execute all evaluations concurrently
            llm_results = await asyncio.gather(*tasks, return_exceptions=True)
            
        
        # Process LLM evaluation results
        reranked_matches = []
        for candidate, llm_result in zip(high_score_candidates, llm_results):
            if isinstance(llm_result, Exception):
                # On LLM evaluation error, preserve original match with warning flag
                print(f"LLM evaluation failed for candidate: {llm_result}")
                candidate["llm_verified"] = False
                candidate["llm_error"] = str(llm_result)
                reranked_matches.append(candidate)
            elif llm_result.get("is_similar", False):
                # LLM confirmed similarity - enhance match with LLM data
                candidate["llm_score"] = llm_result.get("confidence", 0.5)
                candidate["llm_reasoning"] = llm_result.get("reasoning", "")
                candidate["llm_verified"] = True
                reranked_matches.append(candidate)
            # Note: Candidates where LLM determined no similarity are filtered out
        
        # Sort by combined embedding and LLM confidence scores
        reranked_matches.sort(
            key=lambda x: (x["score"] + x.get("llm_score", 0)) / 2,
            reverse=True
        )
        
        return {
            **matches,
            "matches": reranked_matches
        }
    
    async def rerank_matches_batch(self, matches_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple match queries in batches for efficient LLM evaluation.
        
        Uses batch API calls to reduce latency and API costs while maintaining
        accuracy through parallel processing of multiple query-candidate pairs.
        
        Args:
            matches_list: List of match dictionaries to re-rank
            
        Returns:
            List of re-ranked match dictionaries with LLM verification
        """
        if not matches_list:
            return []
        
        # Extract queries that have matches to process
        valid_queries = [m for m in matches_list if m.get("matches")]
        if not valid_queries:
            return matches_list
        
        # Prepare all query-candidate pairs for batch processing
        return await self._prepare_and_process_batch(valid_queries, matches_list)
    
    async def _prepare_and_process_batch(
        self, 
        valid_queries: List[Dict[str, Any]], 
        matches_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare and process query-candidate pairs in batch.
        
        Args:
            valid_queries: Queries that have matches to process
            matches_list: Original list of all matches
            
        Returns:
            Processed matches with LLM evaluation results
        """
        all_pairs = []
        query_candidate_map = {}  # Track which pairs belong to which query
        
        # Build comprehensive mapping of all query-candidate pairs
        for query_idx, query_matches in enumerate(valid_queries):
            query_code = query_matches["code"]
            candidates = [
                c for c in query_matches["matches"]
                if self.similarity_threshold <= c["score"] < 1
            ]
            
            # Create pairs for batch processing
            for cand_idx, candidate in enumerate(candidates):
                pair_id = f"{query_idx}.{cand_idx}"
                all_pairs.append((query_code, candidate))
                query_candidate_map[pair_id] = {
                    'query_idx': query_idx,
                    'candidate_idx': cand_idx,
                    'candidate': candidate
                }
        
        if not all_pairs:
            return matches_list
        
        # Execute batch LLM evaluation
        try:
            all_llm_results = self.sap_ai_core.evaluate_code_similarity_claude_batch(all_pairs, self.model)
        except Exception as e:
            print(f"Batch evaluation failed: {e}")
            # Fallback to individual processing if batch fails
            return await self._fallback_individual_processing(valid_queries)
        
        # Process batch results back to query format
        processed_queries = self._process_batch_results(valid_queries, all_llm_results)
        
        # Return results in original order
        return self._merge_results_with_original(matches_list, processed_queries)
    
    def _process_batch_results(
        self, 
        valid_queries: List[Dict[str, Any]], 
        all_llm_results: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """Process batch LLM results back into query format.
        
        Args:
            valid_queries: Original queries that were processed
            all_llm_results: Results from batch LLM evaluation
            
        Returns:
            Dictionary mapping query indices to processed results
        """
        processed_queries = {}
        pair_idx = 0
        
        for query_idx, query_matches in enumerate(valid_queries):
            candidates = [
                c for c in query_matches["matches"] 
                if c["score"] >= self.similarity_threshold 
            ]
            filtered_matches = []
            
            # Process each candidate with its LLM result
            for candidate in candidates:
                if pair_idx < len(all_llm_results):
                    llm_result = all_llm_results[pair_idx]
                    pair_idx += 1
                    
                    # Include only LLM-verified similar matches
                    if llm_result.get("is_similar", False):
                        candidate["llm_score"] = llm_result.get("confidence", 0.5)
                        candidate["llm_reasoning"] = llm_result.get("reasoning", "")
                        candidate["llm_verified"] = True
                        filtered_matches.append(candidate)
            
            # Sort by combined embedding and LLM scores
            filtered_matches.sort(
                key=lambda x: (x["score"] + x.get("llm_score", 0)) / 2,
                reverse=True
            )
            
            processed_queries[query_idx] = {
                **query_matches,
                "matches": filtered_matches
            }
            
        return processed_queries
    
    def _merge_results_with_original(
        self, 
        matches_list: List[Dict[str, Any]], 
        processed_queries: Dict[int, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge processed results back with original matches list.
        
        Args:
            matches_list: Original list of all matches
            processed_queries: Processed query results
            
        Returns:
            Complete results list in original order
        """
        result = []
        valid_idx = 0
        
        # Preserve original order while including processed results
        for original_matches in matches_list:
            if original_matches.get("matches"):
                result.append(processed_queries.get(valid_idx, original_matches))
                valid_idx += 1
            else:
                result.append(original_matches)
        
        return result
    
    async def _fallback_individual_processing(self, valid_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback to individual processing when batch evaluation fails.
        
        Args:
            valid_queries: Queries to process individually
            
        Returns:
            List of individually processed results
        """
        print("Batch processing failed, falling back to individual processing...")
        results = []
        
        for query_matches in valid_queries:
            try:
                result = await self.rerank_matches(query_matches)
                results.append(result)
            except Exception as e:
                print(f"Individual processing also failed: {e}")
                # Return original matches if both batch and individual processing fail
                results.append(query_matches)
        
        return results
    
    def _evaluate_similarity_sync(self, query_code: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for LLM evaluation (for ThreadPoolExecutor).
        
        This method provides a synchronous interface for async LLM evaluation
        to be used within ThreadPoolExecutor for controlled concurrency.
        
        Args:
            query_code: Source code to compare
            candidate: Candidate match dictionary with code and metadata
            
        Returns:
            Dictionary containing similarity evaluation results
        """
        return asyncio.run(self._evaluate_similarity(query_code, candidate))
    
    async def _evaluate_similarity(self, query_code: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Perform async LLM evaluation of code similarity.
        
        Uses the centralized AI service to evaluate semantic similarity
        between query and candidate code snippets.
        
        Args:
            query_code: Source code snippet to compare against
            candidate: Candidate match containing code and metadata
            
        Returns:
            Dictionary with evaluation results including similarity flag,
            confidence score, and reasoning
        """
        candidate_code = candidate["code"]
        return await self.sap_ai_core.evaluate_code_similarity_claude(
            query_code, 
            candidate_code
        )