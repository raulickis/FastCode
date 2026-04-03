"""
Code Embedder - Generate embeddings for code snippets
"""

import logging
import platform
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class CodeEmbedder:
    """Generate embeddings for code using sentence transformers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_config = config.get("embedding", {})
        self.logger = logging.getLogger(__name__)

        self.model_name = self.embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.device = self.embedding_config.get("device", "auto")
        self.batch_size = self.embedding_config.get("batch_size", 32)
        self.max_seq_length = self.embedding_config.get("max_seq_length", 512)
        self.normalize = self.embedding_config.get("normalize_embeddings", True)

        # Auto-detect best available device: CUDA > MPS > CPU
        if self.device != "cpu":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.logger.info(f"Loading embedding model: {self.model_name}")
        self.model = self._load_model()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.logger.info(f"Embedding dimension: {self.embedding_dim}")

    def _load_model(self) -> SentenceTransformer:
        """Load sentence transformer model"""
        # Arquitetura nativa: sem necessidade de trust_remote_code=True ou monkey-patches
        model = SentenceTransformer(self.model_name, device=self.device)
        model.max_seq_length = self.max_seq_length
        return model

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts

        Args:
            texts: List of input texts

        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])

        encode_kwargs = {
            'batch_size': self.batch_size,
            'show_progress_bar': len(texts) > 100,
            'normalize_embeddings': self.normalize,
            'convert_to_numpy': True,
            'device': self.device,
            'convert_to_tensor': False,
        }

        if platform.system() == 'Darwin':
            encode_kwargs['pool'] = None

        embeddings = self.model.encode(texts, **encode_kwargs)

        return embeddings

    def embed_code_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for code elements (functions, classes, etc.)

        Args:
            elements: List of code element dictionaries

        Returns:
            List of elements with embeddings added
        """
        from .utils import chunk_text, count_tokens

        if not elements:
            return []

        all_texts = []
        element_text_ranges = []

        # Prepare texts
        for elem in elements:
            header = self._prepare_code_header(elem)
            code = elem.get("code", "")

            start_idx = len(all_texts)

            if code:
                # Truncate extremely long codes to prevent infinite chunking issues
                if len(code) > 200000:
                    code = code[:200000] + "..."

                full_text = f"{header}\nCode:\n{code}"
                token_threshold = self.max_seq_length - 100

                if count_tokens(full_text) <= token_threshold:
                    all_texts.append(full_text)
                else:
                    chunks = chunk_text(code, chunk_size=300, overlap=50)
                    if not chunks:
                        all_texts.append(header)
                    else:
                        for chunk in chunks:
                            chunk_content = f"{header}\nCode:\n{chunk['text']}"
                            all_texts.append(chunk_content)
            else:
                all_texts.append(header)

            end_idx = len(all_texts)
            element_text_ranges.append((start_idx, end_idx))

        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(all_texts)} chunks from {len(elements)} code elements")
        all_embeddings = self.embed_batch(all_texts)
        self.logger.info(f"✓ Successfully generated embeddings for {len(all_embeddings)} chunks")

        # Add embeddings to elements
        for i, elem in enumerate(elements):
            start, end = element_text_ranges[i]
            if start < end:
                elem_embeddings = all_embeddings[start:end]
                elem_texts = all_texts[start:end]

                elem["embeddings"] = elem_embeddings
                elem["embedding_texts"] = elem_texts
                # Compatibility properties (using the first chunk)
                elem["embedding"] = elem_embeddings[0]
                elem["embedding_text"] = elem_texts[0]

        return elements

    def _prepare_code_header(self, element: Dict[str, Any]) -> str:
        """
        Prepare code block header containing contextual metadata
        """
        parts = []

        if "type" in element:
            parts.append(f"Type: {element['type']}")

        if "name" in element:
            parts.append(f"Name: {element['name']}")

        if "signature" in element:
            parts.append(f"Signature: {element['signature']}")

        if "docstring" in element and element["docstring"]:
            parts.append(f"Documentation: {element['docstring']}")

        if "summary" in element and element["summary"]:
            parts.append(element["summary"])

        return "\n".join(parts)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        if self.normalize:
            # Already normalized, just dot product
            return float(np.dot(embedding1, embedding2))
        else:
            # Compute cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def compute_similarities(self, query_embedding: np.ndarray,
                            embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarities between query and multiple embeddings

        Args:
            query_embedding: Query embedding vector
            embeddings: Array of embedding vectors

        Returns:
            Array of similarity scores
        """
        if self.normalize:
            # Simple dot product for normalized embeddings
            similarities = np.dot(embeddings, query_embedding)
        else:
            # Compute cosine similarities
            norms = np.linalg.norm(embeddings, axis=1)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return np.zeros(len(embeddings))
            similarities = np.dot(embeddings, query_embedding) / (norms * query_norm)

        return similarities

