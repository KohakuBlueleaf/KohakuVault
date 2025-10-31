"""
Type stubs for the _kvault Rust extension module.

This file provides type hints for the compiled PyO3 extension,
enabling IDE autocomplete and static type checking.
"""

from typing import BinaryIO, Optional

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
