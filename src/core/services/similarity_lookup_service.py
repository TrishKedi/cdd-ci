import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
class SimilarityLookup:
    def __init__(self, similarity_threshold: float = 0.85) -> None:
        """Initialize similarity lookup with threshold configuration.

        Args:
            similarity_threshold: Minimum similarity score for matches
        """
        self.similarity_threshold = similarity_threshold
        self.index = None

    def build_index(self, embeddings):
        dimensions = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimensions)
        self.index.add(embeddings)

    def run_semantic_search(
        self, 
        query_embeddings: np.ndarray, 
   
    ) -> List[Tuple[List[int], List[float]]]:
        """Perform semantic search using FAISS index.

        Args:
            query_embeddings: Query vectors for similarity search
            faiss_index: FAISS index to search against
            
        Returns:
            List of tuples containing (neighbor_ids, similarity_scores)
        """
        similarity_scores, neighbours = self.index.search(query_embeddings, k=3)
        search_results = list(zip(neighbours.tolist(), similarity_scores.tolist()))
        return search_results

    def process_results(self, search_results, query, cand):
        all_matches = []

        for query_id, score_nb in enumerate(search_results):
            neighbours = score_nb[0]
            scores = score_nb[1]
            neighbour_similarity_map = list(zip(neighbours, scores))

            print(f"VECTOR_ID: \n\n\n{query_id}\n\n\n")
            

            block = query[query_id]
            # block_id = block[0]
            # vector_id = block[1]
            path = block.get('path')
            code = block.get('code')
            start = block.get('start')

            matches = {
                'code': code,
                'path': path,
                'start': start,
                'matches':[ 
                    {
                        # "id": code_blocks[ns_map[0]].get('id'),
                        "path": cand[ns_map[0]].get('path'),
                        # query_blocks[ns_map[0]][2], #To do ----extract this from DB via queries
                        "score": ns_map[1],
                        "code": cand[ns_map[0]].get('code'), #To do ----extract this from DB via queries
                        "start": cand[ns_map[0]].get('start'),

                    }
                    for ns_map in neighbour_similarity_map 
                    if ns_map[1] >= self.similarity_threshold
                    # and self.is_unique_match(unique_matches, [vector_id, ns_map[0]])
                    ] 
            }
            # print(matches)
            all_matches.append(matches)

        return all_matches
