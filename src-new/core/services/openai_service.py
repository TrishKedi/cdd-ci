import json
import faiss
import numpy as np
from openai import OpenAI
from typing import List

class OpenAIService:

    def __init__(self, model="text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model

    def generate_embeddings(self, code_batch: List[str]):
        # print(code_batch)
        
        response = self.client.embeddings.create(input=code_batch, model=self.model)
        print("=====Response=======")
        # print(response)

        # embeddings = response.data[0].embedding
        print("=====Embeddings=======")
        embeddings = np.vstack([np.asarray(d.embedding, dtype="float32") for d in response.data])

        print("=====EMBEDDINGS SHAPE=====")
        print(embeddings.shape)
        return embeddings

    def create_index(self, embeddings):
        index = faiss.IndexFlatIP(dimensions)
        index.add(embeddings)

        return index