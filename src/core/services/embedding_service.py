"""FAISS embedding index management for code similarity search.

This module provides the EmbeddingIndex class for creating, loading,
and managing FAISS indexes that store code embeddings for efficient
similarity search operations.
"""

import os
from typing import Optional
import faiss
import numpy as np


class EmbeddingIndex:
    """Manages FAISS indexes for storing and searching code embeddings.
    
    This class provides functionality to create, load, persist, and add
    embeddings to FAISS indexes for efficient similarity search.
    """

    def __init__(self) -> None:
        """Initialize the embedding index with a file path.
     
        """
   
        self.index: Optional[faiss.Index] = None
        self.is_setup = False

    def get_index(self):
        return self.index

    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """Add embeddings to the FAISS index.
        
        Args:
            embeddings: Numpy array of embedding vectors to add
        """
        dimensions = embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Initialize index if not already loaded
        if self.index is None:
            # Create new inner product index for normalized vectors
            self.index = faiss.IndexFlatIP(dimensions)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Persist changes to disk
        # self.persist_index()


