"""
Type stubs for the _kvault Rust extension module.

This file provides type hints for the compiled PyO3 extension,
enabling IDE autocomplete and static type checking.
"""

from typing import Any, BinaryIO, Optional

class _KVault:
    """
    Low-level Rust implementation of KVault.

    This is the compiled PyO3 class. Users should use the `KVault` proxy instead.
    """

    def __init__(
        self,
        path: str,
        table: str = "kv",
        chunk_size: int = 1048576,
        enable_wal: bool = True,
        page_size: int = 32768,
        mmap_size: int = 268435456,
        cache_kb: int = 100000,
        read_only: bool = False,
        reader_cache_kb: int = 8192,
        reader_idle_limit: int = 8,
    ) -> None:
        """
        Initialize a new KVault database.

        Parameters
        ----------
        path : str
            Path to SQLite database file.
        table : str, default="kv"
            Name of the table to use.
        chunk_size : int, default=1048576
            Default chunk size for streaming operations in bytes.
        enable_wal : bool, default=True
            Enable SQLite Write-Ahead Logging.
        page_size : int, default=32768
            SQLite page size (only affects new databases).
        mmap_size : int, default=268435456
            Memory-mapped I/O size in bytes.
        cache_kb : int, default=100000
            SQLite cache size in kilobytes.
        """
        ...

    def enable_cache(
        self,
        cap_bytes: int = 67108864,
        flush_threshold: int = 16777216,
        _flush_interval: float | None = None,
    ) -> None:
        """
        Enable write-back cache for batching writes.

        Parameters
        ----------
        cap_bytes : int, default=67108864
            Maximum cache capacity in bytes.
        flush_threshold : int, default=16777216
            Flush cache when this size is reached.
        """
        ...

    def finish_close(self) -> None:
        """Fence new mutations and flush; failure leaves the handle open."""
        ...

    def cache_stats(self) -> tuple[int, int, int]:
        """Return pending entries, in-flight entries, and total buffered bytes."""
        ...

    def clear(self) -> None: ...
    def pop(self, key: bytes | str) -> Any: ...
    def setdefault(self, key: bytes | str, default: Any = None) -> Any: ...
    def reader_pool_stats(self) -> tuple[int, int]: ...
    def put_many(self, items: list[tuple[bytes | str, Any]] | tuple) -> None: ...
    def get_many(self, keys: list[bytes | str] | tuple, default: Any = None) -> list[Any]: ...
    def disable_cache(self) -> None:
        """Disable write-back cache."""
        ...

    def put(self, key: bytes, value: Any) -> None:
        """
        Store a value for a key.

        Parameters
        ----------
        key : bytes
            Key as bytes.
        value : bytes
            Value as bytes.

        Raises
        ------
        RuntimeError
            If a database error occurs.
        """
        ...

    def put_stream(self, key: bytes, reader: BinaryIO, size: int, chunk_size: int) -> None:
        """
        Stream a value from a file-like object.

        Parameters
        ----------
        key : bytes
            Key as bytes.
        reader : BinaryIO
            File-like object with read() method.
        size : int
            Total size of data to read in bytes.
        chunk_size : int
            Chunk size for reading.

        Raises
        ------
        RuntimeError
            If a database or I/O error occurs.
        """
        ...

    def get(self, key: bytes) -> Any:
        """
        Retrieve value for a key.

        Parameters
        ----------
        key : bytes
            Key as bytes.

        Returns
        -------
        bytes
            Value as bytes.

        Raises
        ------
        RuntimeError
            If key not found or database error occurs.
        """
        ...

    def get_to_file(self, key: bytes, writer: BinaryIO, chunk_size: int) -> int:
        """
        Stream value to a file-like object.

        Parameters
        ----------
        key : bytes
            Key as bytes.
        writer : BinaryIO
            File-like object with write() method.
        chunk_size : int
            Chunk size for writing.

        Returns
        -------
        int
            Number of bytes written.

        Raises
        ------
        RuntimeError
            If key not found or I/O error occurs.
        """
        ...

    def delete(self, key: bytes) -> bool:
        """
        Delete a key.

        Parameters
        ----------
        key : bytes
            Key as bytes.

        Returns
        -------
        bool
            True if key was deleted, False if it didn't exist.
        """
        ...

    def exists(self, key: bytes) -> bool:
        """
        Check if a key exists.

        Parameters
        ----------
        key : bytes
            Key as bytes.

        Returns
        -------
        bool
            True if key exists, False otherwise.
        """
        ...

    def scan_keys(
        self,
        prefix: bytes | None = None,
        limit: int | None = None,
        after: bytes | None = None,
    ) -> list[bytes]: ...
    def scan_items(
        self,
        prefix: bytes | None = None,
        limit: int | None = None,
        after: bytes | None = None,
    ) -> list[tuple[bytes, Any]]: ...
    def checkpoint_wal(self) -> int: ...
    def set_cache_locked(self, locked: bool) -> None: ...
    def enable_auto_pack(self, use_pickle: bool = True) -> None: ...
    def disable_auto_pack(self) -> None: ...
    def auto_pack_enabled(self) -> bool: ...
    def enable_headers(self) -> None: ...
    def disable_headers(self) -> None: ...
    def headers_enabled(self) -> bool: ...
    def flush_cache(self) -> int:
        """
        Flush write-back cache to disk.

        Returns
        -------
        int
            Number of entries flushed.
        """
        ...

    def optimize(self) -> None:
        """
        Optimize and vacuum the database.

        This reclaims space and optimizes the database structure.
        Can be slow for large databases.
        """
        ...

    def len(self) -> int:
        """
        Return the number of keys in the vault.

        Returns
        -------
        int
            Number of keys.
        """
        ...

class DataPacker:
    """
    Rust-based data packer for efficient serialization.

    Supports primitives (i64, f64, str, bytes), MessagePack, and CBOR.
    Provides significant performance improvements for bulk operations.
    """

    def __init__(self, dtype: str) -> None:
        """
        Create a DataPacker from dtype string.

        Parameters
        ----------
        dtype : str
            Data type specification:
            - "i64" - 64-bit signed integer
            - "f64" - 64-bit float
            - "str:utf8" - Variable-size UTF-8 string
            - "str:32" - Fixed 32-byte UTF-8 string (padded)
            - "str:32:utf16le" - Fixed 32-byte UTF-16LE string
            - "str:ascii" - Variable-size ASCII string
            - "str:latin1" - Variable-size Latin-1 string
            - "bytes" - Variable-size raw bytes
            - "bytes:128" - Fixed 128-byte data (padded)
            - "msgpack" - MessagePack (structured data)
            - "cbor" - CBOR (alternative structured format)

        Examples
        --------
        >>> packer = DataPacker("i64")
        >>> packer = DataPacker("str:32:utf8")
        >>> packer = DataPacker("msgpack")
        """
        ...

    @staticmethod
    def with_json_schema(schema: dict[str, Any]) -> "DataPacker":
        """
        Create MessagePack packer with JSON Schema validation.

        Parameters
        ----------
        schema : dict
            JSON Schema definition for validation.

        Returns
        -------
        DataPacker
            Packer with validation enabled.

        Examples
        --------
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "integer", "minimum": 0}
        ...     },
        ...     "required": ["name", "age"]
        ... }
        >>> packer = DataPacker.with_json_schema(schema)
        >>> packer.pack({"name": "Alice", "age": 30})
        """
        ...

    @staticmethod
    def with_cddl_schema(schema: Optional[str] = None) -> "DataPacker":
        """
        Create CBOR packer with optional CDDL schema.

        Parameters
        ----------
        schema : str | None, optional
            CDDL schema string for validation (not yet implemented).

        Returns
        -------
        DataPacker
            CBOR packer instance.
        """
        ...

    def pack(self, value: Any) -> bytes:
        """
        Pack a single value to bytes.

        Parameters
        ----------
        value : Any
            Value to pack (type depends on dtype).

        Returns
        -------
        bytes
            Packed binary data.

        Raises
        ------
        ValueError
            If value is invalid for dtype or validation fails.
        TypeError
            If value type doesn't match dtype.

        Examples
        --------
        >>> packer = DataPacker("i64")
        >>> packed = packer.pack(42)
        >>> len(packed)
        8
        """
        ...

    def pack_many(self, values: list[Any]) -> bytes:
        """
        Pack multiple values to concatenated bytes.

        Variable-width results require offsets when calling unpack_many().

        Parameters
        ----------
        values : list
            List of values to pack.

        Returns
        -------
        bytes
            Concatenated packed data.

        Raises
        ------
        ValueError
            If called on variable-size type.

        Examples
        --------
        >>> packer = DataPacker("i64")
        >>> packed = packer.pack_many([1, 2, 3, 4, 5])
        >>> len(packed)
        40
        """
        ...

    def unpack(self, data: bytes, offset: int = 0) -> Any:
        """
        Unpack a single value from bytes at offset.

        Parameters
        ----------
        data : bytes
            Packed binary data.
        offset : int, default=0
            Byte offset to start unpacking.

        Returns
        -------
        Any
            Unpacked value (type depends on dtype).

        Raises
        ------
        ValueError
            If not enough data or invalid format.

        Examples
        --------
        >>> packer = DataPacker("i64")
        >>> packed = packer.pack(42)
        >>> packer.unpack(packed, 0)
        42
        """
        ...

    def unpack_many(
        self, data: bytes, count: int | None = None, offsets: list[int] | None = None
    ) -> list[Any]:
        """
        Unpack multiple values from bytes.

        For fixed-size types: Use count parameter.
        For variable-size types: Use offsets parameter.

        Parameters
        ----------
        data : bytes
            Packed binary data.
        count : int | None
            Number of values to unpack (required for fixed-size types).
        offsets : list[int] | None
            Start offsets of each element (required for variable-size types).

        Returns
        -------
        list
            List of unpacked values.

        Raises
        ------
        ValueError
            If wrong parameter for type or not enough data.

        Examples
        --------
        >>> # Fixed-size type
        >>> packer = DataPacker("i64")
        >>> packed = packer.pack_many([1, 2, 3])
        >>> packer.unpack_many(packed, count=3)
        [1, 2, 3]

        >>> # Variable-size type
        >>> packer = DataPacker("str:utf8")
        >>> strings = ["hello", "world"]
        >>> packed = packer.pack_many(strings)
        >>> offsets = [0, 5]  # "hello" starts at 0, "world" at 5
        >>> packer.unpack_many(packed, offsets=offsets)
        ['hello', 'world']
        """
        ...

    @property
    def elem_size(self) -> int:
        """
        Get element size in bytes (0 for variable-size types).

        Returns
        -------
        int
            Size in bytes, or 0 if variable-size.

        Examples
        --------
        >>> DataPacker("i64").elem_size
        8
        >>> DataPacker("msgpack").elem_size
        0
        """
        ...

    @property
    def is_varsize(self) -> bool:
        """
        Check if this is a variable-size type.

        Returns
        -------
        bool
            True if variable-size, False if fixed-size.

        Examples
        --------
        >>> DataPacker("i64").is_varsize
        False
        >>> DataPacker("msgpack").is_varsize
        True
        """
        ...

from ._kvault_csbtree import CSBTree
from ._kvault_skiplist import SkipList

class _ColumnVault:
    def __init__(self, path: str, read_only: bool = False) -> None: ...
    def create(self, name: str, dtype: str, elem_size: int, block_bytes: int) -> int: ...
    def info(self, name: str) -> tuple[int, str, int, int, int]: ...
    def length(self, id: int) -> int: ...
    def list_columns(self) -> list[tuple[str, str, int]]: ...
    def delete_column(self, name: str) -> bool: ...
    def clear(self, id: int) -> None: ...
    def close(self) -> None: ...
    def checkpoint_wal(self) -> tuple[int, int, int]: ...
    def append_fixed(self, id: int, data: bytes) -> None: ...
    def read_fixed(self, id: int, start: int = 0, count: int | None = None) -> bytes: ...
    def take_fixed(self, id: int, indices: list[int]) -> bytes: ...
    def overwrite_fixed(self, id: int, start: int, data: bytes) -> None: ...
    def append_variable(self, id: int, data: bytes, offsets: bytes) -> None: ...
    def read_variable(
        self, id: int, start: int = 0, count: int | None = None
    ) -> tuple[bytes, bytes]: ...
    def replace_fixed(self, id: int, data: bytes) -> None: ...
    def replace_variable(self, id: int, data: bytes, offsets: bytes) -> None: ...
    def storage_info(self, id: int) -> tuple[int, int, int]: ...

class _VectorKVault:
    def __init__(
        self,
        path: str,
        table: str = "vec_kvault",
        dimensions: int = 768,
        metric: str = "cosine",
        vector_type: str = "f32",
        read_only: bool = False,
    ) -> None: ...
    def insert(self, vector: Any, value: Any, metadata: dict | None = None) -> int: ...
    def search(
        self, query_vector: Any, k: int = 10, metric: str | None = None
    ) -> list[tuple[int, float, Any]]: ...
    def get(self, query_vector: Any, metric: str | None = None) -> Any: ...
    def get_by_id(self, id: int) -> tuple[bytes, Any]: ...
    def delete(self, id: int) -> None: ...
    def update(self, id: int, vector: Any = None, value: Any = None) -> None: ...
    def exists(self, id: int) -> bool: ...
    def count(self) -> int: ...
    def info(self) -> dict[str, Any]: ...
    def enable_auto_pack(self, use_pickle: bool = True) -> None: ...
    def disable_auto_pack(self) -> None: ...
    def auto_pack_enabled(self) -> bool: ...
    def enable_headers(self) -> None: ...
    def disable_headers(self) -> None: ...
    def headers_enabled(self) -> bool: ...
