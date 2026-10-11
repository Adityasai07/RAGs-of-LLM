from typing import Any, List, Dict
from src.vector_store import VectorStore
from src.embeddings import EmbeddingManager

class RAGretriver:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrive(self, query: str, topk: int = 5, threshold_score: float = 0.0) -> List[Dict[str, Any]]:
        print(f"query: {query}\ttopk: {topk}\tThreshold score: {threshold_score}")
        embeddings_ofquery = self.embedding_manager.generate_embeddings([query])[0]
        try:
            results_ofquery_from_vector_store = self.vector_store.collection.query(
                query_embeddings=[embeddings_ofquery.tolist()],
                n_results=topk
            )
            retrived_documents = []
            if results_ofquery_from_vector_store["documents"] and results_ofquery_from_vector_store["documents"][0]:
                documents = results_ofquery_from_vector_store["documents"][0]
                metadatas = results_ofquery_from_vector_store["metadatas"][0]
                distances = results_ofquery_from_vector_store["distances"][0]
                ids = results_ofquery_from_vector_store["ids"][0]

                for document_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
                    score = 1.0 - distance
                    if score > threshold_score:
                        retrived_documents.append({
                            "id": document_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": score,
                            "distance": distance,
                            "rank": len(retrived_documents) + 1
                        })
                print(f"Retrieved documents count: {len(retrived_documents)}")
            else:
                print("No relevant docs found")

            return retrived_documents
        except Exception as e:
            print(f"Error occurred: {e}")
            raise