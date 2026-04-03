"""
Vector Store - Store and retrieve code embeddings using Qdrant
"""

import os
import pickle
import logging
import uuid
from re import X
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from filelock import FileLock, Timeout

from qdrant_client import QdrantClient, models
from .utils import ensure_dir


class VectorStore:
    """Vector database for code embeddings using Qdrant"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_config = config.get("vector_store", {})
        self.logger = logging.getLogger(__name__)

        # Evaluation mode can request a purely in-memory index that never touches disk.
        self.in_memory = self.vector_config.get(
            "in_memory",
            config.get("evaluation", {}).get("in_memory_index", False),
        )
        # Keep repo overviews in-memory when persistence is disabled.
        self._in_memory_repo_overviews: Dict[str, Dict[str, Any]] = {}

        self.dimension = None
        self.index = None # Kept as None to avoid breaking legacy checks, though unused by Qdrant
        self.metadata = []  # Store metadata for each vector

        self.persist_dir = self.vector_config.get("persist_directory", "./data/vector_store")
        self.distance_metric = self.vector_config.get("distance_metric", "cosine")
        self.index_type = self.vector_config.get("index_type", "HNSW")

        # HNSW parameters
        self.m = self.vector_config.get("m", 16)
        self.ef_construction = self.vector_config.get("ef_construction", 200)
        self.ef_search = self.vector_config.get("ef_search", 50)

        # Qdrant configuration
        self.qdrant_url = self.vector_config.get("url", "http://localhost:6333")
        self.collection_name = self.vector_config.get("collection_name", "fastcode_elements")
        self.overview_collection = self.vector_config.get("overview_collection", "fastcode_overviews")

        # Initialize Qdrant Client
        if self.in_memory:
            self.client = QdrantClient(location=":memory:")
            self.logger.info("Qdrant initialized in in-memory mode; persistence disabled.")
        else:
            self.client = QdrantClient(url=self.qdrant_url)
            self.logger.info(f"Qdrant client connected to {self.qdrant_url}")

        # Cache for scan_available_indexes to avoid repeated file I/O
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
        Initialize the vector store

        Args:
            dimension: Dimension of embedding vectors
        """
        self.dimension = dimension
        self.logger.info(f"Initializing vector store with dimension {dimension}")

        collections_to_create = [self.collection_name, self.overview_collection]

        for coll in collections_to_create:
            if not self.client.collection_exists(collection_name=coll):
                self.client.create_collection(
                    collection_name=coll,
                    vectors_config=models.VectorParams(
                        size=dimension,
                        distance=self._get_distance_metric()
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=self.m,
                        ef_construct=self.ef_construction
                    )
                )
                self.logger.info(f"Qdrant collection '{coll}' created.")

                # Native Pre-filtering Indices (Resolving Gap 3)
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
        self.logger.info(f"Initialized Qdrant {self.index_type} index with {self.distance_metric} distance")

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add vectors to the store

        Args:
            vectors: Array of embedding vectors (N x dimension)
            metadata: List of metadata dictionaries for each vector
        """
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")

        # Ensure vectors are float32
        vectors = vectors.astype(np.float32)

        points = []
        for vec, meta in zip(vectors, metadata):
            # Deterministic UUID mapping for Qdrant
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
        if points:
            self.client.upload_points(
                collection_name=self.collection_name,
                points=points
            )

        self.metadata.extend(metadata)

        self.logger.info(f"Added {len(vectors)} vectors to store (total: {len(self.metadata)})")

    def search(self, query_vector: np.ndarray, k: int = 10,
               min_score: Optional[float] = None, repo_filter: Optional[List[str]] = None,
               element_type_filter: Optional[str] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            min_score: Minimum similarity score (optional)
            repo_filter: Optional list of repository names to filter by
            element_type_filter: Optional element type to filter by (e.g., "repository_overview")

        Returns:
            List of (metadata, score) tuples
        """
        if len(self.metadata) == 0:
            return []

        # Ensure query is float32 and 2D
        query_vector = query_vector.astype(np.float32).reshape(1, -1)

        must_conditions = []

        # Native Qdrant Payload Pre-filtering
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

        # Search with Qdrant
        search_k = min(k, len(self.metadata))
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.flatten().tolist(),
            limit=search_k,
            query_filter=query_filter,
            score_threshold=min_score
        )

        results = []
        for hit in hits:
            # For Qdrant, Cosine distance directly returns similarity score 0-1
            score = hit.score
            results.append((hit.payload, score))

        return results

    def save_repo_overview(self, repo_name: str, overview_content: str,
                          embedding: np.ndarray, metadata: Dict[str, Any]):
        """
        Save a single repository overview to a separate file and Qdrant

        Args:
            repo_name: Name of the repository
            overview_content: Text content of the overview
            embedding: Embedding vector for the overview
            metadata: Additional metadata (repo_url, summary, structure, etc.)
        """
        if self.in_memory:
            # Keep entirely in memory during evaluation.
            self._in_memory_repo_overviews[repo_name] = {
                "repo_name": repo_name,
                "content": overview_content,
                "embedding": embedding.astype(np.float32),
                "metadata": metadata,
            }
            self.logger.info(f"Stored repository overview for {repo_name} (in-memory)")
            return

        # 1. Save to Qdrant Native Collection
        point_id = self._get_qdrant_id(f"overview_{repo_name}")
        payload = {
            "repo_name": repo_name,
            "content": overview_content,
            "metadata": metadata
        }
        try:
            self.client.upsert(
                collection_name=self.overview_collection,
                points=[models.PointStruct(id=point_id, vector=embedding.flatten().tolist(), payload=payload)]
            )
        except Exception as e:
            self.logger.error(f"Failed to save overview to Qdrant for {repo_name}: {e}")

        # 2. Save to Disk (Backward compatibility / Web UI support)
        overview_path = os.path.join(self.persist_dir, "repo_overviews.pkl")
        lock_path = f"{overview_path}.lock"

        # --- THREAD/PROCESS SAFE BLOCK ---
        try:
            with FileLock(lock_path, timeout=15):
                # Load existing overviews if they exist
                overviews = {}
                if os.path.exists(overview_path):
                    try:
                        with open(overview_path, 'rb') as f:
                            overviews = pickle.load(f)
                    except Exception as e:
                        self.logger.warning(f"Failed to load existing repo overviews: {e}")

                # Add/update this repository's overview
                overviews[repo_name] = {
                    "repo_name": repo_name,
                    "content": overview_content,
                    "embedding": embedding.astype(np.float32),
                    "metadata": metadata
                }

                # Save back to file
                with open(overview_path, 'wb') as f:
                    pickle.dump(overviews, f)

                self.logger.info(f"Saved repository overview for {repo_name}")
        except Timeout:
            self.logger.error(f"Timeout waiting for lock on {overview_path}")
        except Exception as e:
            self.logger.error(f"Failed to save repository overview: {e}")

    def delete_repo_overview(self, repo_name: str) -> bool:
        """
        Delete a repository overview from storage

        Args:
            repo_name: Name of the repository to remove

        Returns:
            True if the overview was found and removed
        """
        if self.in_memory:
            if repo_name in self._in_memory_repo_overviews:
                del self._in_memory_repo_overviews[repo_name]
                self.logger.info(f"Deleted in-memory overview for {repo_name}")
                return True
            return False

        # 1. Delete from Qdrant
        try:
            self.client.delete(
                collection_name=self.overview_collection,
                points_selector=models.Filter(
                    must=[models.FieldCondition(key="repo_name", match=models.MatchValue(value=repo_name))]
                )
            )
        except Exception as e:
            self.logger.error(f"Failed to delete repo overview from Qdrant: {e}")

        # 2. Delete from Disk
        overview_path = os.path.join(self.persist_dir, "repo_overviews.pkl")
        if not os.path.exists(overview_path):
            return False

        lock_path = f"{overview_path}.lock"

        # --- THREAD/PROCESS SAFE BLOCK ---
        try:
            with FileLock(lock_path, timeout=10):
                with open(overview_path, 'rb') as f:
                    overviews = pickle.load(f)

                if repo_name not in overviews:
                    return False

                del overviews[repo_name]

                with open(overview_path, 'wb') as f:
                    pickle.dump(overviews, f)

                self.logger.info(f"Deleted repository overview for {repo_name}")
                return True
        except Timeout:
            self.logger.error(f"Timeout waiting for lock on {overview_path}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete repository overview for {repo_name}: {e}")
            return False

    def load_repo_overviews(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all repository overviews from storage

        Returns:
            Dictionary mapping repo_name to overview data
        """
        if self.in_memory:
            # Return the in-memory overviews when persistence is disabled.
            return self._in_memory_repo_overviews

        overview_path = os.path.join(self.persist_dir, "repo_overviews.pkl")

        if not os.path.exists(overview_path):
            self.logger.info("No repository overviews found")
            return {}

        lock_path = f"{overview_path}.lock"

        # --- THREAD SAFE READ (Prevents reading while another process is writing) ---
        try:
            with FileLock(lock_path, timeout=10):
                with open(overview_path, 'rb') as f:
                    overviews = pickle.load(f)
            self.logger.info(f"Loaded {len(overviews)} repository overviews")
            return overviews
        except Timeout:
            self.logger.error(f"Timeout waiting for lock to read {overview_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load repository overviews: {e}")
            return {}

    def search_repository_overviews(self, query_vector: np.ndarray, k: int = 5,
                                    min_score: Optional[float] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search specifically for repository overview elements using native Qdrant search

        Args:
            query_vector: Query embedding vector
            k: Number of repositories to return
            min_score: Minimum similarity score

        Returns:
            List of (metadata, score) tuples for repository overviews only
        """
        # Native Qdrant Search on Overview Collection
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
        """
        Search for multiple queries at once natively in Qdrant

        Args:
            query_vectors: Array of query vectors (N x dimension)
            k: Number of results per query
            min_score: Minimum similarity score

        Returns:
            List of result lists (one per query)
        """
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
        """Get number of vectors in store"""
        return len(self.metadata)

    def get_repository_names(self) -> List[str]:
        """Get list of unique repository names in the store"""
        repo_names = set()
        for meta in self.metadata:
            repo_name = meta.get("repo_name")
            if repo_name:
                repo_names.add(repo_name)
        return sorted(list(repo_names))

    def get_count_by_repository(self) -> Dict[str, int]:
        """Get count of vectors per repository"""
        repo_counts = {}
        for meta in self.metadata:
            repo_name = meta.get("repo_name", "unknown")
            repo_counts[repo_name] = repo_counts.get(repo_name, 0) + 1
        return repo_counts

    def filter_by_repositories(self, repo_names: List[str]) -> List[int]:
        """
        Get indices of vectors belonging to specific repositories

        Args:
            repo_names: List of repository names to filter by

        Returns:
            List of indices
        """
        indices = []
        for i, meta in enumerate(self.metadata):
            if meta.get("repo_name") in repo_names:
                indices.append(i)
        return indices

    def save(self, name: str = "index"):
        """
        Save metadata to disk to keep UI and incremental indexing functional.
        FAISS export is skipped since Qdrant manages vector persistence natively.

        Args:
            name: Name for the saved files
        """
        if self.in_memory:
            self.logger.info("Skipping vector store save (in-memory mode enabled)")
            return

        metadata_path = os.path.join(self.persist_dir, f"{name}_metadata.pkl")
        lock_path = os.path.join(self.persist_dir, f"{name}.lock")

        try:
            with FileLock(lock_path, timeout=30):
                # Save metadata (Shadow copy for FastCode mechanisms)
                with open(metadata_path, 'wb') as f:
                    pickle.dump({
                        "metadata": self.metadata,
                        "dimension": self.dimension,
                        "distance_metric": self.distance_metric,
                        "index_type": "Qdrant",
                    }, f)

            # Invalidate cache since we just modified the indexes
            self.invalidate_scan_cache()
            self.logger.info(f"Saved metadata to {metadata_path}")

        except Timeout:
            self.logger.error(f"Timeout waiting for lock to save index '{name}'")
        except Exception as e:
            self.logger.error(f"Failed to save vector index metadata '{name}': {e}")

    def load(self, name: str = "index") -> bool:
        """
        Load metadata from disk. Vectors are already in Qdrant, we just repopulate memory metadata.

        Args:
            name: Name of the saved files

        Returns:
            True if successful, False otherwise
        """
        if self.in_memory:
            self.logger.info("Skipping vector store load (in-memory mode enabled)")
            return False

        metadata_path = os.path.join(self.persist_dir, f"{name}_metadata.pkl")

        if not os.path.exists(metadata_path):
            self.logger.warning(f"Index files not found in {self.persist_dir}")
            return False

        try:
            # Load metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data.get("metadata", [])
                self.dimension = data.get("dimension", self.dimension)
                self.distance_metric = data.get("distance_metric", "cosine")
                self.index_type = data.get("index_type", "Qdrant")

            # Ensure collections exist in Qdrant
            if self.dimension:
                self.initialize(self.dimension)

            self.logger.info(
                f"Loaded vector store metadata with {len(self.metadata)} items "
                f"from {self.persist_dir}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to load vector store metadata: {e}")
            return False

    def clear(self):
        """Clear all vectors from Qdrant and reset metadata"""
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
        In a unified Qdrant DB, merging just means loading the metadata
        for the UI to know it's in scope. Vectors are already in Qdrant.

        Args:
            index_name: Name of the index to merge from

        Returns:
            True if successful, False otherwise
        """
        if self.in_memory:
            self.logger.info("Skipping merge_from_index (in-memory mode enabled)")
            return False

        metadata_path = os.path.join(self.persist_dir, f"{index_name}_metadata.pkl")

        if not os.path.exists(metadata_path):
            self.logger.warning(f"Index files not found for {index_name}")
            return False

        try:
            # Load metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                other_metadata = data.get("metadata", [])

            # Avoid duplicating metadata in memory
            existing_ids = {m.get("id") for m in self.metadata if m.get("id")}
            for meta in other_metadata:
                if meta.get("id") not in existing_ids:
                    self.metadata.append(meta)

            self.logger.info(f"Merged metadata from {index_name} (vectors natively maintained by Qdrant)")
            return True

        except Exception as e:
            self.logger.error(f"Failed to merge from {index_name}: {e}")
            return False

    def delete_by_filter(self, filter_func) -> int:
        """
        Delete vectors matching a filter function

        Args:
            filter_func: Function that takes metadata and returns True to delete

        Returns:
            Number of vectors deleted
        """
        indices_to_keep = []
        metadata_to_keep = []

        for i, meta in enumerate(self.metadata):
            if not filter_func(meta):
                indices_to_keep.append(i)
                metadata_to_keep.append(meta)

        num_deleted = len(self.metadata) - len(metadata_to_keep)

        if num_deleted > 0:
            self.logger.info(f"Updating memory metadata after removing {num_deleted} vectors")
            self.metadata = metadata_to_keep

            self.logger.warning(
                "Note: Qdrant physical point deletion via arbitrary Python filter_func is not natively supported here. "
                "The metadata was updated, but points may still reside in the Vector DB."
            )

        return num_deleted

    def scan_available_indexes(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Scan persist directory for available index metadata files (with caching)

        Args:
            use_cache: Use cached results if available (default: True)

        Returns:
            List of dictionaries with repository information
        """
        import time

        available_repos = []

        if self.in_memory:
            self.logger.info("Skipping index scan (in-memory mode enabled)")
            return available_repos

        if not os.path.exists(self.persist_dir):
            return available_repos

        # Check cache
        if use_cache and self._index_scan_cache is not None:
            cache_time, cached_results = self._index_scan_cache
            if time.time() - cache_time < self._index_scan_cache_ttl:
                self.logger.debug("Using cached index scan results")
                return cached_results

        # Perform actual scan
        self.logger.info("Scanning available indexes...")

        for file in os.listdir(self.persist_dir):
            # FIX: We now scan for _metadata.pkl instead of .faiss since FAISS is removed
            if file.endswith('_metadata.pkl') and file != 'repo_overviews.pkl':
                repo_name = file.replace('_metadata.pkl', '')
                metadata_file = os.path.join(self.persist_dir, file)

                if os.path.exists(metadata_file):
                    try:
                        # Get file sizes (fast operation)
                        metadata_size = os.path.getsize(metadata_file)
                        total_size_mb = metadata_size / (1024 * 1024)

                        # Optimized: Only read first chunk of metadata for basic info
                        # This avoids loading potentially huge metadata files
                        element_count = 0
                        file_count = 0
                        repo_url = "N/A"

                        with open(metadata_file, 'rb') as f:
                            try:
                                data = pickle.load(f)
                                metadata_list = data.get("metadata", [])
                                element_count = len(metadata_list)

                                # Sample first few entries to get URL and estimate file count
                                # (much faster than iterating through all)
                                sample_size = min(self._index_scan_sample_size, len(metadata_list))
                                seen_files = set()

                                for i in range(sample_size):
                                    meta = metadata_list[i]
                                    file_path = meta.get("file_path")
                                    if file_path:
                                        seen_files.add(file_path)
                                    if not repo_url or repo_url == "N/A":
                                        repo_url = meta.get("repo_url", "N/A")

                                # Estimate total file count based on sample
                                if sample_size > 0 and sample_size < len(metadata_list):
                                    file_count = int(len(seen_files) * (len(metadata_list) / sample_size))
                                else:
                                    file_count = len(seen_files)

                            except Exception as load_error:
                                self.logger.warning(f"Failed to parse metadata for {repo_name}: {load_error}")

                        available_repos.append({
                            'name': repo_name,
                            'element_count': element_count,
                            'file_count': file_count,
                            'size_mb': round(total_size_mb, 2),
                            'url': repo_url,
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to read metadata for {repo_name}: {e}")
                        # Still add it with minimal info
                        available_repos.append({
                            'name': repo_name,
                            'element_count': 0,
                            'file_count': 0,
                            'size_mb': 0,
                            'url': "N/A",
                        })

        results = sorted(available_repos, key=lambda x: x['name'])

        # Update cache
        self._index_scan_cache = (time.time(), results)
        self.logger.info(f"Index scan complete: found {len(results)} repositories")

        return results

    def invalidate_scan_cache(self):
        """Invalidate the scan cache (call this when indexes change)"""
        self._index_scan_cache = None
        self.logger.debug("Invalidated index scan cache")
