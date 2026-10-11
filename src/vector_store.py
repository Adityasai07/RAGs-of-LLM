import os
import uuid
from typing import List, Any
import numpy as np
import chromadb

class VectorStore:
    def __init__(self, collection_name: str = "multi_format_docs", db_directory: str = "data/VectorStore"):
        self.collection_name = collection_name
        self.db_directory = db_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.db_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.db_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "model_name": "all-MiniLM-L6-v2",
                    "description": "Multi-format document RAG collection"
                }
            )
            print("Collection:", self.collection_name)
            print("Collections count:", self.collection.count())
        except Exception as e:
            raise ValueError(f"Error Occurred! {e}")

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError(f"Chunks and embeddings length mismatch: {len(embeddings)} vs {len(documents)}")

        print(f"Preparing to add {len(embeddings)} docs...")
        ids, new_embeddings, new_metadatas, new_documents = [], [], [], []

        for i, (document, embedding) in enumerate(zip(documents, embeddings)):
            document_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(document_id)

            m = dict(document.metadata)
            m["doc_index"] = i
            m["content_length"] = len(document.page_content)

            new_metadatas.append(m)
            new_documents.append(document.page_content)
            new_embeddings.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=new_embeddings,
                metadatas=new_metadatas,
                documents=new_documents
            )
            print("Successfully added to collection!")
            print("Collections Count:", self.collection.count())
        except Exception as e:
            raise ValueError(f"Error Occurred! {e}")

    def count(self) -> int:
        return self.collection.count()