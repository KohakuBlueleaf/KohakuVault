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

    def enable_cache(self, cap_bytes: int = 67108864, flush_threshold: int = 16777216) -> None:
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

    def disable_cache(self) -> None:
        """Disable write-back cache."""
        ...

    def put(self, key: bytes, value: bytes) -> None:
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

    def get(self, key: bytes) -> bytes:
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

    def scan_keys(self, prefix: Optional[bytes] = None, limit: int = 1000) -> list[bytes]:
        """
        Scan keys, optionally with a prefix filter.

        Parameters
        ----------
        prefix : bytes | None
            If provided, only return keys starting with this prefix.
        limit : int, default=1000
            Maximum number of keys to return.

        Returns
        -------
        list[bytes]
            List of keys as bytes.
        """
        ...

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

        Only works for fixed-size types (i64, f64, fixed-size str/bytes).

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
