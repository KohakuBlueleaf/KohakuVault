import io
import os
import time
import threading
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Optional, Union
from collections.abc import Mapping
from contextlib import contextmanager

from kohakuvault import errors as E
from kohakuvault._kvault import _KVault  # compiled PyO3 class


# Type aliases for better readability
KeyType = Union[bytes, str]
ValueType = Union[bytes, bytearray, memoryview]
PathLike = Union[str, Path, os.PathLike]


# ----------------------------
# Helpers
# ----------------------------


def _to_bytes_key(key: KeyType) -> bytes:
    """Normalize key to bytes."""
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode("utf-8")
    raise E.InvalidArgument("key must be bytes or str")


def _is_bytes_like(obj: Any) -> bool:
    """Check if object is bytes-like."""
    return isinstance(obj, (bytes, bytearray, memoryview))


def _is_file_like(obj: Any) -> bool:
    """Check if object has read() method."""
    return hasattr(obj, "read")


def _is_writer_like(obj: Any) -> bool:
    """Check if object has write() method."""
    return hasattr(obj, "write")


def _stream_copy(reader: BinaryIO, writer: BinaryIO, chunk_size: int = 1 << 20) -> int:
    """Copy data from reader to writer in chunks."""
    total = 0
    while True:
        chunk = reader.read(chunk_size)
        if not chunk:
            break
        n = writer.write(chunk)
        if n is None:
            n = len(chunk)
        total += n
    return total


def _file_size_of(f: BinaryIO) -> Optional[int]:
    """Try to determine remaining bytes in file-like object."""
    try:
        # If real file, use fileno/stat for speed
        if hasattr(f, "fileno"):
            try:
                st = os.fstat(f.fileno())
                # If it's a regular file, we can use st_size and current position
                if os.path.isfile(getattr(f, "name", "")) or (st.st_mode & 0o170000) == 0o100000:
                    pos = f.tell()
                    remaining = max(0, st.st_size - pos)
                    return remaining
            except OSError:
                pass
        # Generic seek/tell
        pos = f.tell()
        f.seek(0, os.SEEK_END)
        end = f.tell()
        f.seek(pos, os.SEEK_SET)
        return max(0, end - pos)
    except Exception:
        return None


def _with_retries(
    call, *, attempts: int, backoff_base: float, key_for_error: Optional[bytes] = None
):
    """Retry transient busy/locked errors with exponential backoff."""
    delay = backoff_base
    last_exc = None
    for i in range(attempts):
        try:
            return call()
        except Exception as ex:
            mapped = E.map_exception(ex, key=key_for_error)
            last_exc = mapped
            if isinstance(mapped, E.DatabaseBusy) and i < attempts - 1:
                time.sleep(delay)
                delay *= 2.0
                continue
            raise mapped
    # Should not reach here, but in case:
    if last_exc:
        raise last_exc
    raise RuntimeError("Retry logic failed without exception")  # pragma: no cover


# ----------------------------
# Public API
# ----------------------------


class KVault(Mapping):
    """
    High-level, Pythonic wrapper around the Rust `_KVault`.

    Provides a dict-like interface for storing binary blobs in SQLite.
    Optimized for large media files with streaming support.

    Examples
    --------
    Basic dict-like usage:
        >>> vault = KVault("data.db")
        >>> vault["key"] = b"value"
        >>> vault["key"]
        b'value'
        >>> "key" in vault
        True
        >>> del vault["key"]
        >>> len(vault)
        0

    Context manager:
        >>> with KVault("data.db") as vault:
        ...     vault["key"] = b"data"
        ...     # Automatic flush on exit

    Streaming large files:
        >>> with open("video.mp4", "rb") as f:
        ...     vault.put_file("video", f)
        >>> with open("output.mp4", "wb") as f:
        ...     vault.get_to_file("video", f)

    Iteration:
        >>> for key in vault:
        ...     print(key)
        >>> for key in vault.keys(prefix=b"img_"):
        ...     print(key)

    Parameters
    ----------
    path : str | Path | os.PathLike
        SQLite database file path.
    chunk_size : int, default=1_048_576 (1 MiB)
        Default chunk size for streaming operations.
    retries : int, default=4
        Number of retry attempts for transient SQLite busy/locked errors.
    backoff_base : float, default=0.02
        Initial backoff delay in seconds for retries (exponentially increases).
    table : str, default="kv"
        SQLite table name to use.
    enable_wal : bool, default=True
        Enable SQLite WAL (Write-Ahead Logging) mode.
    page_size : int, default=32768
        SQLite page size in bytes (only effective on new databases).
    mmap_size : int, default=268_435_456 (256 MiB)
        Memory-mapped I/O size.
    cache_kb : int, default=100_000 (100 MB)
        SQLite cache size in kilobytes.
    """

    def __init__(
        self,
        path: PathLike,
        *,
        chunk_size: int = 1 << 20,  # 1 MiB
        retries: int = 4,
        backoff_base: float = 0.02,
        table: str = "kv",
        enable_wal: bool = True,
        page_size: int = 32768,
        mmap_size: int = 268_435_456,
        cache_kb: int = 100_000,
    ):
        self._chunk = int(chunk_size)
        self._retries = int(retries)
        self._backoff = float(backoff_base)
        self._closed = False
        self._daemon_thread = None
        self._daemon_stop = threading.Event()

        try:
            db_path = os.fspath(path)
        except Exception as ex:
            raise E.InvalidArgument(f"Invalid path: {path!r}") from ex

        try:
            self._inner = _KVault(
                db_path,
                table=table,
                chunk_size=chunk_size,
                enable_wal=enable_wal,
                page_size=page_size,
                mmap_size=mmap_size,
                cache_kb=cache_kb,
            )
        except Exception as ex:
            raise E.map_exception(ex)

        self._path = db_path

    def __repr__(self) -> str:
        """Return string representation."""
        status = "closed" if self._closed else "open"
        try:
            count = len(self) if not self._closed else "?"
        except Exception:
            count = "?"
        return f"KVault({self._path!r}, status={status}, keys={count})"

    # ----------------------------
    # Context Manager Protocol
    # ----------------------------

    def __enter__(self) -> _KVault:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, flushing cache."""
        self.close()

    def close(self) -> None:
        """Flush cache, checkpoint WAL, stop daemon thread, and mark as closed."""
        if not self._closed:
            try:
                self._stop_daemon_thread()
                self.flush_cache()
                # Checkpoint WAL to main DB file
                self.checkpoint()
            finally:
                self._closed = True

    def checkpoint(self) -> None:
        """
        Manually checkpoint WAL file to main database.

        This merges the WAL file into the main DB file, preventing
        the WAL from growing indefinitely. Called automatically on close(),
        but can be called manually for long-running processes.
        """
        if self._closed:
            return
        try:
            self._inner.checkpoint_wal()
        except Exception:
            pass  # Ignore checkpoint errors (non-critical)

    # ----------------------------
    # Dict-like Interface
    # ----------------------------

    def __getitem__(self, key: KeyType) -> bytes:
        """Get value by key. Raises KeyError if not found."""
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")

        k = _to_bytes_key(key)

        def call():
            return self._inner.get(k)

        try:
            return _with_retries(
                call, attempts=self._retries, backoff_base=self._backoff, key_for_error=k
            )
        except E.NotFound:
            raise KeyError(key)

    def __setitem__(self, key: KeyType, value: ValueType) -> None:
        """Set value for key."""
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")
        self.put(key, value)

    def __delitem__(self, key: KeyType) -> None:
        """Delete key. Raises KeyError if not found."""
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")
        if not self.delete(key):
            raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        """Check if key exists."""
        if self._closed:
            return False
        try:
            return self.exists(key)
        except Exception:
            return False

    def __len__(self) -> int:
        """Return number of keys."""
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")
        try:
            return int(self._inner.len())
        except Exception as ex:
            raise E.map_exception(ex)

    def __iter__(self) -> Iterator[bytes]:
        """Iterate over all keys."""
        return self.keys()

    # ----------------------------
    # Mapping Protocol
    # ----------------------------

    def keys(self, prefix: Optional[KeyType] = None, limit: int = 10000) -> Iterator[bytes]:
        """
        Iterate over keys, optionally with a prefix filter.

        Parameters
        ----------
        prefix : bytes | str | None
            If provided, only return keys starting with this prefix.
        limit : int, default=10000
            Maximum number of keys to return per batch.

        Yields
        ------
        bytes
            Key as bytes.
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")

        prefix_bytes = _to_bytes_key(prefix) if prefix is not None else None

        def call():
            return self._inner.scan_keys(prefix_bytes, limit)

        try:
            keys_list = _with_retries(
                call, attempts=self._retries, backoff_base=self._backoff, key_for_error=prefix_bytes
            )
            yield from keys_list
        except Exception as ex:
            raise E.map_exception(ex, key=prefix_bytes)

    def values(self) -> Iterator[bytes]:
        """
        Iterate over all values.

        Warning: This loads each value into memory. Not recommended for large blobs.
        """
        for key in self.keys():
            yield self.get(key)

    def items(self) -> Iterator[tuple[bytes, bytes]]:
        """
        Iterate over all (key, value) pairs.

        Warning: This loads each value into memory. Not recommended for large blobs.
        """
        for key in self.keys():
            yield key, self.get(key)

    def get(self, key: KeyType, default: Any = None) -> Any:
        """
        Get value for key, returning default if not found.

        Parameters
        ----------
        key : bytes | str
            Key to retrieve.
        default : Any, optional
            Value to return if key not found.

        Returns
        -------
        bytes | Any
            Value as bytes, or default if not found.
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")

        k = _to_bytes_key(key)

        def call():
            return self._inner.get(k)

        try:
            return _with_retries(
                call, attempts=self._retries, backoff_base=self._backoff, key_for_error=k
            )
        except E.NotFound:
            return default
        except E.KohakuVaultError:
            raise

    # ----------------------------
    # Core Operations
    # ----------------------------

    def put(self, key: KeyType, value: ValueType) -> None:
        """
        Store a value for a key.

        Parameters
        ----------
        key : bytes | str
            Key to store under.
        value : bytes | bytearray | memoryview
            Value to store (must be bytes-like).
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")

        k = _to_bytes_key(key)
        if not _is_bytes_like(value):
            raise E.InvalidArgument("value must be bytes-like")

        def call():
            return self._inner.put(k, bytes(value))

        _with_retries(call, attempts=self._retries, backoff_base=self._backoff, key_for_error=k)

    def put_file(
        self,
        key: KeyType,
        reader: Union[BinaryIO, ValueType],
        *,
        size: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        """
        Store value by streaming from a file-like object or bytes.

        Use this for large files to avoid loading everything into memory.

        Parameters
        ----------
        key : bytes | str
            Key to store under.
        reader : BinaryIO | bytes | bytearray | memoryview
            File-like object with read() method, or bytes-like object.
        size : int | None
            Size of data in bytes. If None, will try to infer from file.
        chunk_size : int | None
            Chunk size for streaming. If None, uses default.

        Examples
        --------
        >>> with open("large_file.bin", "rb") as f:
        ...     vault.put_file("mykey", f)
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")

        k = _to_bytes_key(key)

        # bytes-like fast path
        if _is_bytes_like(reader):
            return self.put(k, reader)

        if not _is_file_like(reader):
            raise E.InvalidArgument("reader must be file-like or bytes-like")

        # Size handling
        inferred = _file_size_of(reader) if size is None else size
        if inferred is None:
            # Fallback: buffer into BytesIO to know size
            try:
                buf = io.BytesIO()
                _stream_copy(reader, buf, chunk_size or self._chunk)
                return self.put(k, buf.getbuffer())
            except Exception as ex:
                raise E.map_exception(ex, key=k)

        size_int = int(inferred)
        ch = int(chunk_size or self._chunk)

        def call():
            return self._inner.put_stream(k, reader, size_int, ch)

        _with_retries(call, attempts=self._retries, backoff_base=self._backoff, key_for_error=k)

    def get_to_file(
        self, key: KeyType, writer: BinaryIO, *, chunk_size: Optional[int] = None
    ) -> int:
        """
        Stream value to a file-like object.

        Use this for large files to avoid loading everything into memory.

        Parameters
        ----------
        key : bytes | str
            Key to retrieve.
        writer : BinaryIO
            File-like object with write() method.
        chunk_size : int | None
            Chunk size for streaming. If None, uses default.

        Returns
        -------
        int
            Number of bytes written.

        Examples
        --------
        >>> with open("output.bin", "wb") as f:
        ...     bytes_written = vault.get_to_file("mykey", f)
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")

        if not _is_writer_like(writer):
            raise E.InvalidArgument("writer must be a file-like object with write()")
        k = _to_bytes_key(key)
        ch = int(chunk_size or self._chunk)

        def call():
            return int(self._inner.get_to_file(k, writer, ch))

        return _with_retries(
            call, attempts=self._retries, backoff_base=self._backoff, key_for_error=k
        )

    def delete(self, key: KeyType) -> bool:
        """
        Delete a key.

        Parameters
        ----------
        key : bytes | str
            Key to delete.

        Returns
        -------
        bool
            True if key was deleted, False if it didn't exist.
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")

        k = _to_bytes_key(key)

        def call():
            return bool(self._inner.delete(k))

        return _with_retries(
            call, attempts=self._retries, backoff_base=self._backoff, key_for_error=k
        )

    def exists(self, key: KeyType) -> bool:
        """
        Check if a key exists.

        Parameters
        ----------
        key : bytes | str
            Key to check.

        Returns
        -------
        bool
            True if key exists, False otherwise.
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")

        k = _to_bytes_key(key)

        def call():
            return bool(self._inner.exists(k))

        return _with_retries(
            call, attempts=self._retries, backoff_base=self._backoff, key_for_error=k
        )

    def pop(self, key: KeyType, default: Any = None) -> Any:
        """
        Remove and return value for key.

        Parameters
        ----------
        key : bytes | str
            Key to remove.
        default : Any, optional
            Value to return if key not found.

        Returns
        -------
        bytes | Any
            Value as bytes, or default if not found.
        """
        try:
            value = self.get(key)
            self.delete(key)
            return value
        except E.NotFound:
            return default

    def setdefault(self, key: KeyType, default: ValueType) -> bytes:
        """
        Get value for key, setting it to default if not present.

        Parameters
        ----------
        key : bytes | str
            Key to retrieve or set.
        default : bytes | bytearray | memoryview
            Value to set if key doesn't exist.

        Returns
        -------
        bytes
            Existing or newly set value.
        """
        if self.exists(key):
            return self.get(key)
        else:
            self.put(key, default)
            return bytes(default)

    def update(self, other: Union[Mapping, Iterator[tuple[KeyType, ValueType]]], **kwargs) -> None:
        """
        Update vault with key-value pairs from another mapping or iterable.

        Parameters
        ----------
        other : Mapping | Iterable
            Mapping or iterable of (key, value) pairs.
        **kwargs
            Additional key-value pairs.
        """
        if isinstance(other, Mapping):
            for key, value in other.items():
                self[key] = value
        else:
            for key, value in other:
                self[key] = value

        for key, value in kwargs.items():
            self[key] = value

    def clear(self) -> None:
        """
        Remove all keys from the vault.

        Warning: This deletes all data!
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")

        for key in list(self.keys()):
            self.delete(key)

    # ----------------------------
    # Cache Management
    # ----------------------------

    def enable_cache(
        self,
        *,
        cap_bytes: int = 64 << 20,
        flush_threshold: int = 16 << 20,
        flush_interval: Optional[float] = None,
    ) -> None:
        """
        Enable write-back cache to batch writes.

        Parameters
        ----------
        cap_bytes : int, default=67_108_864 (64 MiB)
            Maximum cache size in bytes.
        flush_threshold : int, default=16_777_216 (16 MiB)
            Flush cache when this size is reached.
        flush_interval : float, optional
            If provided, start daemon thread that auto-flushes after this many
            seconds of idle time. Example: 5.0 = flush after 5 seconds of no writes.
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")
        try:
            self._inner.enable_cache(cap_bytes, flush_threshold, flush_interval)
        except Exception as ex:
            raise E.map_exception(ex)

        # Start daemon thread if interval specified
        if flush_interval is not None:
            self._start_daemon_thread(flush_interval)

    def disable_cache(self) -> None:
        """Disable write-back cache and stop daemon thread if running.

        IMPORTANT: Flushes cache before disabling to prevent data loss!
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")

        # Stop daemon thread first
        self._stop_daemon_thread()

        # Flush any remaining cached data before disabling
        self.flush_cache()

        try:
            self._inner.disable_cache()
        except Exception as ex:
            raise E.map_exception(ex)

    def flush_cache(self) -> int:
        """
        Flush write-back cache to disk.

        Returns
        -------
        int
            Number of entries flushed.
        """
        if self._closed:
            return 0
        try:
            return int(self._inner.flush_cache())
        except Exception as ex:
            raise E.map_exception(ex)

    @contextmanager
    def cache(
        self,
        cap_bytes: int = 64 << 20,
        flush_threshold: Optional[int] = None,
        auto_flush: bool = True,
    ):
        """
        Context manager for scoped write-back caching.

        Automatically flushes cache on exit (even if exception occurs).

        Parameters
        ----------
        cap_bytes : int, default=67_108_864 (64 MiB)
            Maximum cache size in bytes.
        flush_threshold : int, optional
            Flush when this size reached. Defaults to cap_bytes // 4.
        auto_flush : bool, default=True
            If True, automatically flush on exit. If False, user must flush manually.

        Examples
        --------
        >>> with vault.cache(64 * 1024 * 1024):
        ...     for i in range(10000):
        ...         vault[f"key:{i}"] = data
        ... # Auto-flushed here, guaranteed!

        >>> with vault.cache(auto_flush=False):
        ...     vault["key"] = data
        ...     # auto_flush=False skips flush in finally block
        ...     # but disable_cache() still flushes for safety
        """
        if flush_threshold is None:
            flush_threshold = cap_bytes // 4

        self.enable_cache(cap_bytes=cap_bytes, flush_threshold=flush_threshold)
        try:
            yield self
        finally:
            if auto_flush:
                self.flush_cache()
            self.disable_cache()

    @contextmanager
    def lock_cache(self):
        """
        Context manager to prevent auto-flush during critical section.

        Useful for atomic multi-key operations when using daemon thread.

        Examples
        --------
        >>> vault.enable_cache(flush_interval=5.0)
        >>> with vault.lock_cache():
        ...     vault["config:part1"] = data1
        ...     vault["config:part2"] = data2
        ... # Auto-flush deferred until here
        """
        self._inner.set_cache_locked(True)
        try:
            yield
        finally:
            self._inner.set_cache_locked(False)

    def _start_daemon_thread(self, flush_interval: float) -> None:
        """Start background thread for auto-flushing cache after idle period."""
        # Stop existing daemon if any
        self._stop_daemon_thread()

        self._daemon_stop.clear()

        def daemon_worker():
            last_flush = time.time()
            while not self._daemon_stop.wait(0.5):  # Check every 500ms
                if self._closed:
                    break

                try:
                    # Check if enough time has passed since last flush
                    if time.time() - last_flush >= flush_interval:
                        # Flush cache if not empty
                        flushed = self.flush_cache()
                        if flushed > 0:
                            last_flush = time.time()
                except Exception:
                    pass  # Ignore errors in daemon

        self._daemon_thread = threading.Thread(target=daemon_worker, daemon=True)
        self._daemon_thread.start()

    def _stop_daemon_thread(self) -> None:
        """Stop the daemon thread if running."""
        if self._daemon_thread and self._daemon_thread.is_alive():
            self._daemon_stop.set()
            self._daemon_thread.join(timeout=2.0)
            self._daemon_thread = None

    # ----------------------------
    # Maintenance
    # ----------------------------

    def optimize(self) -> None:
        """
        Optimize and vacuum the database.

        This reclaims space and optimizes the database structure.
        Can be slow for large databases.
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")
        try:
            self._inner.optimize()
        except Exception as ex:
            raise E.map_exception(ex)
