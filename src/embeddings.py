import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print("Started to load model:", self.model_name)
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully:", self.model_name)
        except Exception as e:
            print(f"Error occurred in _load_model: {e}")
            raise

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded yet!")
        print("Length of texts:", len(texts))
        print("Started Embedding!")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

    def get_dimensions(self):
        print(f"Dimensions: {self.model.get_embedding_dimension()}")