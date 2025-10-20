import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional


class VectorDatabase:
    """
    Manages the vector database. Includes text + code
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "trajectories",
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initializes the vector store, loading text embedding models.
        """
        self.text_model = SentenceTransformer(text_model_name)

        self.text_dim = self.text_model.get_sentence_embedding_dimension()

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def _embed(self, trajectory: Optional[str]) -> List[float]:
        """
        Embeds the given trajectory using the text embedding model.
        """
        if trajectory:
            embedding = self.text_model.encode(trajectory, normalize_embeddings=True)
            return embedding.tolist()
        else:
            return np.zeros(self.text_dim, dtype=np.float32).tolist()

    def add(self, trajectory_id: str, trajectory: str, metadata: Dict[str, Any]):
        """
        Adds a trajectory to the vector database with its embedding and metadata.
        """
        embedding = self._embed(trajectory)

        metadata["trajectory_text"] = trajectory

        self.collection.upsert(
            documents=[trajectory],
            embeddings=[embedding],
            ids=[trajectory_id],
            metadatas=[metadata],
        )

    def search(
        self,
        trajectory: str,
        n_results: int,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Searches for similar trajectories in the vector database.
        """

        embedding = self._embed(trajectory)

        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where_filter,
        )
