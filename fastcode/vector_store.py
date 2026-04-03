"""
Vector Store - Store and retrieve code embeddings using Qdrant
"""

import os
import pickle
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from filelock import FileLock, Timeout

from qdrant_client import QdrantClient, models
from .utils import ensure_dir


class VectorStore:
    """Vector database for code embeddings using Qdrant (Native Pre-filtering)"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_config = config.get("vector_store", {})
        self.logger = logging.getLogger(__name__)

        self.in_memory = self.vector_config.get(
            "in_memory",
            config.get("evaluation", {}).get("in_memory_index", False),
        )
        self._in_memory_repo_overviews: Dict[str, Dict[str, Any]] = {}

        self.dimension = None
        self.metadata = []  # Kept for backward compatibility with main.py incremental indexing

        # Qdrant configuration
        self.qdrant_url = self.vector_config.get("url", "http://localhost:6333")
        self.collection_name = self.vector_config.get("collection_name", "fastcode_elements")
        self.overview_collection = self.vector_config.get("overview_collection", "fastcode_overviews")
        self.distance_metric = self.vector_config.get("distance_metric", "cosine")
        
        # Initialize Qdrant Client
        if self.in_memory:
            self.client = QdrantClient(location=":memory:")
            self.logger.info("Qdrant initialized in-memory mode")
        else:
            self.client = QdrantClient(url=self.qdrant_url)
            self.logger.info(f"Qdrant client connected to {self.qdrant_url}")

        self.persist_dir = self.vector_config.get("persist_directory", "./data/vector_store")
        
        # Cache for scan_available_indexes
        self._index_scan_cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None
        self._index_scan_cache_ttl = self.vector_config.get("index_scan_cache_ttl", 30.0)
        self._index_scan_sample_size = self.vector_config.get("index_scan_sample_size", 100)

        if not self.in_memory:
            ensure_dir(self.persist_dir)

    def _get_qdrant_id(self, string_id: str) -> str:
        """Qdrant requires UUIDs or Unsigned Integers. We hash the string ID deterministically."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

    def _get_distance_metric(self):
        if self.distance_metric == "cosine":
            return models.Distance.COSINE
        elif self.distance_metric == "euclidean":
            return models.Distance.EUCLID
        return models.Distance.DOT

    def initialize(self, dimension: int):
        """
        Initialize the vector store collections in Qdrant
        """
        self.dimension = dimension
        self.logger.info(f"Initializing Qdrant collections with dimension {dimension}")

        collections = [self.collection_name, self.overview_collection]
        
        for coll in collections:
            if not self.client.collection_exists(collection_name=coll):
                self.client.create_collection(
                    collection_name=coll,
                    vectors_config=models.VectorParams(
                        size=dimension,
                        distance=self._get_distance_metric()
                    ),
                    # Otimizacoes HNSW nativas
                    hnsw_config=models.HnswConfigDiff(
                        m=self.vector_config.get("m", 16),
                        ef_construct=self.vector_config.get("ef_construct", 100)
                    )
                )
                self.logger.info(f"Collection '{coll}' created.")
                
                # Criar indices de payload para busca exata (O GAP 3 MORRE AQUI)
                if coll == self.collection_name:
                    self.client.create_payload_index(
                        collection_name=coll,
                        field_name="repo_name",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                    self.client.create_payload_index(
                        collection_name=coll,
                        field_name="type",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )

        self.metadata = []

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add vectors to Qdrant

        Args:
            vectors: Array of embedding vectors (N x dimension)
            metadata: List of metadata dictionaries for each vector
        """
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")

        points = []
        for vec, meta in zip(vectors, metadata):
            # Resolve ID (Fallback to random UUID if element lacks ID)
            original_id = meta.get("id", str(uuid.uuid4()))
            chunk_idx = meta.get("chunk_idx", 0)
            point_id = self._get_qdrant_id(f"{original_id}_{chunk_idx}")

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload=meta
                )
            )

        # Batch upsert to Qdrant
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points
        )
        
        # Keep metadata in memory for backward compatibility with save/load .pkl files
        self.metadata.extend(metadata)
        self.logger.info(f"Upserted {len(points)} points into Qdrant collection '{self.collection_name}'")

    def search(self, query_vector: np.ndarray, k: int = 10,
               min_score: Optional[float] = None, repo_filter: Optional[List[str]] = None,
               element_type_filter: Optional[str] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar vectors natively in Qdrant with Payload Pre-filtering (Solving Gap 3)
        """
        must_conditions = []
        
        # O Pulo do Gato: Pre-filtering nativo em vez de if condicional
        if repo_filter:
            must_conditions.append(
                models.FieldCondition(
                    key="repo_name",
                    match=models.MatchAny(any=repo_filter)
                )
            )
            
        if element_type_filter:
            must_conditions.append(
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=element_type_filter)
                )
            )

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.flatten().tolist(),
            limit=k,
            query_filter=query_filter,
            score_threshold=min_score
        )

        results = []
        for hit in hits:
            # Qdrant scores for Cosine are already 0-1 similarities
            results.append((hit.payload, hit.score))

        return results

    def save_repo_overview(self, repo_name: str, overview_content: str,
                          embedding: np.ndarray, metadata: Dict[str, Any]):
        """Save overview to Qdrant AND .pkl (for UI compat)"""
        if self.in_memory:
            self._in_memory_repo_overviews[repo_name] = {
                "repo_name": repo_name,
                "content": overview_content,
                "embedding": embedding.astype(np.float32),
                "metadata": metadata,
            }
            return

        # 1. Save to Qdrant
        point_id = self._get_qdrant_id(f"overview_{repo_name}")
        payload = {
            "repo_name": repo_name,
            "content": overview_content,
            "metadata": metadata
        }
        self.client.upsert(
            collection_name=self.overview_collection,
            points=[models.PointStruct(id=point_id, vector=embedding.flatten().tolist(), payload=payload)]
        )

        # 2. Save to PKL (Legacy compatibility for UI)
        overview_path = os.path.join(self.persist_dir, "repo_overviews.pkl")
        lock_path = f"{overview_path}.lock"
        try:
            with FileLock(lock_path, timeout=15):
                overviews = {}
                if os.path.exists(overview_path):
                    with open(overview_path, 'rb') as f:
                        overviews = pickle.load(f)
                
                overviews[repo_name] = {
                    "repo_name": repo_name,
                    "content": overview_content,
                    "embedding": embedding.astype(np.float32),
                    "metadata": metadata
                }
                
                with open(overview_path, 'wb') as f:
                    pickle.dump(overviews, f)
        except Exception as e:
            self.logger.error(f"Failed to save repository overview pkl: {e}")

    def delete_repo_overview(self, repo_name: str) -> bool:
        """Delete from Qdrant AND .pkl"""
        if self.in_memory:
            if repo_name in self._in_memory_repo_overviews:
                del self._in_memory_repo_overviews[repo_name]
                return True
            return False

        # 1. Delete from Qdrant
        self.client.delete(
            collection_name=self.overview_collection,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="repo_name", match=models.MatchValue(value=repo_name))]
            )
        )

        # 2. Delete from PKL
        overview_path = os.path.join(self.persist_dir, "repo_overviews.pkl")
        if not os.path.exists(overview_path):
            return False

        lock_path = f"{overview_path}.lock"
        try:
            with FileLock(lock_path, timeout=10):
                with open(overview_path, 'rb') as f:
                    overviews = pickle.load(f)
                if repo_name in overviews:
                    del overviews[repo_name]
                    with open(overview_path, 'wb') as f:
                        pickle.dump(overviews, f)
                    return True
        except Exception as e:
            self.logger.error(f"Failed to delete repo overview from pkl: {e}")
            return False
        return False

    def load_repo_overviews(self) -> Dict[str, Dict[str, Any]]:
        """Load from .pkl to preserve legacy format for UI"""
        if self.in_memory:
            return self._in_memory_repo_overviews

        overview_path = os.path.join(self.persist_dir, "repo_overviews.pkl")
        if not os.path.exists(overview_path):
            return {}

        lock_path = f"{overview_path}.lock"
        try:
            with FileLock(lock_path, timeout=10):
                with open(overview_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load repo overviews pkl: {e}")
            return {}

    def search_repository_overviews(self, query_vector: np.ndarray, k: int = 5,
                                    min_score: Optional[float] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Search specifically for repository overview elements natively in Qdrant"""
        hits = self.client.search(
            collection_name=self.overview_collection,
            query_vector=query_vector.flatten().tolist(),
            limit=k,
            score_threshold=min_score
        )

        results = []
        for hit in hits:
            # Reconstruct the metadata wrapper expected by the legacy API
            result_metadata = {
                "repo_name": hit.payload.get("repo_name"),
                "type": "repository_overview",
                **hit.payload.get("metadata", {})
            }
            results.append((result_metadata, hit.score))

        return results

    def search_batch(self, query_vectors: np.ndarray, k: int = 10,
                     min_score: Optional[float] = None) -> List[List[Tuple[Dict[str, Any], float]]]:
        """Search for multiple queries at once using Qdrant search_batch"""
        if len(query_vectors) == 0:
            return []

        search_requests = [
            models.SearchRequest(
                vector=vec.tolist(),
                limit=k,
                score_threshold=min_score
            ) for vec in query_vectors
        ]

        batch_hits = self.client.search_batch(
            collection_name=self.collection_name,
            requests=search_requests
        )

        all_results = []
        for hits in batch_hits:
            results = [(hit.payload, hit.score) for hit in hits]
            all_results.append(results)

        return all_results

    def get_count(self) -> int:
        """Get total number of elements in current context"""
        return len(self.metadata)

    def get_repository_names(self) -> List[str]:
        """Get unique repository names from memory metadata"""
        repo_names = {meta.get("repo_name") for meta in self.metadata if meta.get("repo_name")}
        return sorted(list(repo_names))

    def get_count_by_repository(self) -> Dict[str, int]:
        repo_counts = {}
        for meta in self.metadata:
            repo_name = meta.get("repo_name", "unknown")
            repo_counts[repo_name] = repo_counts.get(repo_name, 0) + 1
        return repo_counts

    def filter_by_repositories(self, repo_names: List[str]) -> List[int]:
        indices = []
        for i, meta in enumerate(self.metadata):
            if meta.get("repo_name") in repo_names:
                indices.append(i)
        return indices

    def save(self, name: str = "index"):
        """
        Save metadata to disk to keep UI and incremental indexing functional.
        FAISS export is skipped since Qdrant manages vector persistence.
        """
        if self.in_memory:
            return

        metadata_path = os.path.join(self.persist_dir, f"{name}_metadata.pkl")
        lock_path = os.path.join(self.persist_dir, f"{name}.lock")

        try:
            with FileLock(lock_path, timeout=30):
                with open(metadata_path, 'wb') as f:
                    pickle.dump({
                        "metadata": self.metadata,
                        "dimension": self.dimension,
                        "distance_metric": self.distance_metric,
                        "index_type": "Qdrant",
                    }, f)

            self.invalidate_scan_cache()
            self.logger.info(f"Saved metadata (Qdrant shadow) to {metadata_path}")

        except Exception as e:
            self.logger.error(f"Failed to save metadata '{name}': {e}")

    def load(self, name: str = "index") -> bool:
        """
        Load metadata from disk. 
        Vectors are already in Qdrant, we just repopulate memory metadata.
        """
        if self.in_memory:
            return False

        metadata_path = os.path.join(self.persist_dir, f"{name}_metadata.pkl")

        if not os.path.exists(metadata_path):
            return False

        try:
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data.get("metadata", [])
                self.dimension = data.get("dimension", self.dimension)
            
            # Assegurar que as collections existem no Qdrant (caso o container tenha restartado)
            if self.dimension:
                self.initialize(self.dimension)

            self.logger.info(f"Loaded {len(self.metadata)} metadata items for {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return False

    def clear(self):
        """Clear the collection from Qdrant and reset metadata"""
        try:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
            if self.client.collection_exists(self.overview_collection):
                self.client.delete_collection(self.overview_collection)
                
            if self.dimension:
                self.initialize(self.dimension)
        except Exception as e:
            self.logger.error(f"Failed to clear Qdrant collections: {e}")
            
        self.metadata = []
        self.logger.info("Cleared vector store")

    def merge_from_index(self, index_name: str) -> bool:
        """
        In a unified Qdrant DB, "merging" just means loading the metadata
        for the UI to know it's in scope. The vectors are already searchable.
        """
        if self.in_memory:
            return False

        metadata_path = os.path.join(self.persist_dir, f"{index_name}_metadata.pkl")

        if not os.path.exists(metadata_path):
            return False

        try:
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                other_metadata = data.get("metadata", [])
                
            # Avoid duplicating metadata in memory
            existing_ids = {m.get("id") for m in self.metadata if m.get("id")}
            for meta in other_metadata:
                if meta.get("id") not in existing_ids:
                    self.metadata.append(meta)
                    
            self.logger.info(f"Appended metadata for {index_name} (vectors already in Qdrant)")
            return True
        except Exception as e:
            self.logger.error(f"Failed to merge metadata from {index_name}: {e}")
            return False

    def delete_by_filter(self, filter_func) -> int:
        """Not efficiently supported by generic functions in Qdrant; kept for interface parity."""
        self.logger.warning("Generic delete_by_filter is inefficient. Target Qdrant via specific queries.")
        return 0

    def scan_available_indexes(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Reads metadata .pkl files to scan available indexes for the Web UI.
        Works identically to the FAISS implementation.
        """
        import time

        available_repos = []

        if self.in_memory or not os.path.exists(self.persist_dir):
            return available_repos

        if use_cache and self._index_scan_cache is not None:
            cache_time, cached_results = self._index_scan_cache
            if time.time() - cache_time < self._index_scan_cache_ttl:
                return cached_results

        self.logger.info("Scanning available indexes (via PKL shadows)...")

        for file in os.listdir(self.persist_dir):
            if file.endswith('_metadata.pkl') and file != 'repo_overviews.pkl':
                repo_name = file.replace('_metadata.pkl', '')
                metadata_file = os.path.join(self.persist_dir, file)

                try:
                    metadata_size = os.path.getsize(metadata_file)
                    total_size_mb = metadata_size / (1024 * 1024)

                    element_count = 0
                    file_count = 0
                    repo_url = "N/A"

                    with open(metadata_file, 'rb') as f:
                        data = pickle.load(f)
                        metadata_list = data.get("metadata", [])
                        element_count = len(metadata_list)

                        sample_size = min(self._index_scan_sample_size, len(metadata_list))
                        seen_files = set()

                        for i in range(sample_size):
                            meta = metadata_list[i]
                            file_path = meta.get("file_path")
                            if file_path:
                                seen_files.add(file_path)
                            if repo_url == "N/A":
                                repo_url = meta.get("repo_url", "N/A")

                        if sample_size > 0 and sample_size < len(metadata_list):
                            file_count = int(len(seen_files) * (len(metadata_list) / sample_size))
                        else:
                            file_count = len(seen_files)

                    available_repos.append({
                        'name': repo_name,
                        'element_count': element_count,
                        'file_count': file_count,
                        'size_mb': round(total_size_mb, 2),
                        'url': repo_url,
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to read metadata shadow for {repo_name}: {e}")

        results = sorted(available_repos, key=lambda x: x['name'])
        self._index_scan_cache = (time.time(), results)
        return results

    def invalidate_scan_cache(self):
        self._index_scan_cache = None
        self.logger.debug("Invalidated index scan cache")
