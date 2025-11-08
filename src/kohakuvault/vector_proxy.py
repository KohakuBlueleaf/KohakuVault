"""
VectorKVault - Vector similarity search storage

This module provides a KVault-like interface with vector keys and similarity search.
Built on top of sqlite-vec's vec0 virtual table.

Uses numpy arrays as the standard interface for vectors.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ._kvault import _VectorKVault
import kohakuvault.errors as E

VectorInput = Union[np.ndarray, List[float], bytes]


class VectorKVault:
    """Vector similarity search with arbitrary values.

    Provides both single-match (get) and k-NN search interfaces.
    Uses numpy arrays as the standard interface for vectors.

    Args:
        path: Database file path
        table: Table name (default: "vec_kvault")
        dimensions: Vector dimensions (required)
        metric: Distance metric - "cosine", "l2", "l1", or "hamming" (default: "cosine")
        vector_type: Element type - "f32", "int8", or "bit" (default: "f32")

    Examples:
        >>> import numpy as np
        >>> from kohakuvault import VectorKVault
        >>>
        >>> # Create VectorKVault for 768-dim embeddings with cosine similarity
        >>> vkv = VectorKVault("data.db", dimensions=768, metric="cosine")
        >>>
        >>> # Insert vector-value pairs (numpy arrays preferred)
        >>> embedding = np.random.randn(768).astype(np.float32)
        >>> doc_id = vkv.insert(embedding, b"document content")
        >>>
        >>> # Search for k-nearest neighbors
        >>> query = np.random.randn(768).astype(np.float32)
        >>> results = vkv.search(query, k=10)
        >>> for id, distance, value in results:
        >>>     print(f"ID {id}: distance={distance:.4f}")
        >>>
        >>> # Get single most similar (KVault-like interface)
        >>> most_similar = vkv.get(query)
        >>>
        >>> # Get vector and value by ID (returns numpy array)
        >>> vector, value = vkv.get_by_id(doc_id)
        >>> assert isinstance(vector, np.ndarray)
        >>>
        >>> # Update by ID
        >>> new_embedding = np.random.randn(768).astype(np.float32)
        >>> vkv.update(doc_id, vector=new_embedding)
        >>> vkv.update(doc_id, value=b"updated content")
    """

    def __init__(
        self,
        path: str,
        table: str = "vec_kvault",
        dimensions: int = 768,
        metric: str = "cosine",
        vector_type: str = "f32",
    ):
        self._vault = _VectorKVault(path, table, dimensions, metric, vector_type)
        self.path = path
        self.table = table
        self.dimensions = dimensions
        self.metric = metric
        self.vector_type = vector_type

    def _prepare_vector(self, vector: VectorInput) -> List[float]:
        """Convert vector to list format for Rust backend."""
        if isinstance(vector, np.ndarray):
            # Ensure it's float32 or float64, then convert to list
            if not np.issubdtype(vector.dtype, np.floating):
                vector = vector.astype(np.float32)
            return vector.tolist()
        elif isinstance(vector, list):
            return vector
        elif isinstance(vector, bytes):
            # Assume bytes represent float32 array
            num_floats = len(vector) // 4
            arr = np.frombuffer(vector, dtype=np.float32, count=num_floats)
            return arr.tolist()
        else:
            raise TypeError(f"Vector must be numpy array, list, or bytes, got {type(vector)}")

    def _prepare_value(self, value: Any) -> bytes:
        """Convert value to bytes."""
        if isinstance(value, bytes):
            return value
        elif isinstance(value, bytearray):
            return bytes(value)
        elif isinstance(value, str):
            return value.encode("utf-8")
        else:
            raise TypeError(f"Value must be bytes, bytearray, or str, got {type(value)}")

    def insert(
        self,
        vector: VectorInput,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a vector-value pair.

        Args:
            vector: Vector as numpy array (preferred), list of floats, or bytes
            value: Value to store (bytes, str, or bytes-like)
            metadata: Optional dict of metadata (not yet supported)

        Returns:
            ID of inserted item
        """
        try:
            vec = self._prepare_vector(vector)
            val = self._prepare_value(value)
            return self._vault.insert(vec, val, metadata)
        except Exception as ex:
            raise E.map_exception(ex)

    def search(
        self,
        query_vector: VectorInput,
        k: int = 10,
        metric: Optional[str] = None,
    ) -> List[Tuple[int, float, bytes]]:
        """Search for k-nearest neighbors.

        Args:
            query_vector: Query vector as numpy array (preferred), list, or bytes
            k: Number of results to return (default: 10)
            metric: Override distance metric (optional)

        Returns:
            List of (id, distance, value) tuples sorted by distance
        """
        try:
            vec = self._prepare_vector(query_vector)
            return self._vault.search(vec, k, metric)
        except Exception as ex:
            raise E.map_exception(ex)

    def get(self, query_vector: VectorInput, metric: Optional[str] = None) -> bytes:
        """Get value for the most similar vector (KVault-like interface).

        Args:
            query_vector: Query vector as numpy array (preferred), list, or bytes
            metric: Override distance metric (optional)

        Returns:
            Value of the most similar vector

        Raises:
            NotFound: If no vectors found in database
        """
        try:
            vec = self._prepare_vector(query_vector)
            return self._vault.get(vec, metric)
        except Exception as ex:
            raise E.map_exception(ex)

    def get_by_id(self, id: int) -> Tuple[np.ndarray, bytes]:
        """Get vector and value by ID.

        Args:
            id: Row ID

        Returns:
            (vector, value) tuple where vector is a numpy array

        Raises:
            NotFound: If ID doesn't exist
        """
        try:
            vector_bytes, value = self._vault.get_by_id(id)
            # Convert bytes to numpy array
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            return vector, value
        except Exception as ex:
            raise E.map_exception(ex)

    def delete(self, id: int) -> None:
        """Delete by ID.

        Args:
            id: Row ID to delete
        """
        try:
            self._vault.delete(id)
        except Exception as ex:
            raise E.map_exception(ex)

    def update(
        self,
        id: int,
        vector: Optional[VectorInput] = None,
        value: Optional[Any] = None,
    ) -> None:
        """Update vector or value by ID.

        Args:
            id: Row ID
            vector: New vector as numpy array (preferred), list, or bytes (optional)
            value: New value (optional)

        Raises:
            ValueError: If neither vector nor value provided
        """
        try:
            vec = self._prepare_vector(vector) if vector is not None else None
            val = self._prepare_value(value) if value is not None else None
            self._vault.update(id, vec, val)
        except Exception as ex:
            raise E.map_exception(ex)

    def exists(self, id: int) -> bool:
        """Check if ID exists.

        Args:
            id: Row ID

        Returns:
            True if exists
        """
        try:
            return self._vault.exists(id)
        except Exception as ex:
            raise E.map_exception(ex)

    def count(self) -> int:
        """Get total count of vectors.

        Returns:
            Number of vectors in database
        """
        try:
            return self._vault.count()
        except Exception as ex:
            raise E.map_exception(ex)

    def info(self) -> Dict[str, Any]:
        """Get VectorKVault info.

        Returns:
            Dict with table info (dimensions, metric, vector_type, count)
        """
        try:
            return self._vault.info()
        except Exception as ex:
            raise E.map_exception(ex)

    def __len__(self) -> int:
        """Return count of vectors."""
        return self.count()

    def __contains__(self, id: int) -> bool:
        """Check if ID exists."""
        return self.exists(id)

    def __repr__(self) -> str:
        return (
            f"VectorKVault(path={self.path!r}, table={self.table!r}, "
            f"dimensions={self.dimensions}, metric={self.metric!r}, "
            f"vector_type={self.vector_type!r}, count={len(self)})"
        )
