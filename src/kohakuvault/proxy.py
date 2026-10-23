import io
import os
import operator
import threading
import time
import warnings
import math
from itertools import chain, islice
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Optional, Union

import kohakuvault.errors as E
from kohakuvault._kvault import _KVault  # compiled PyO3 class

# Type aliases for better readability
KeyType = Union[bytes, str]
ValueType = Union[bytes, bytearray, memoryview]
PathLike = Union[str, Path, os.PathLike]
_MISSING = object()


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


def _stream_size(value: Any, name: str, *, positive: bool = False) -> int:
    """Validate byte counts without silently truncating non-integers."""
    try:
        value = operator.index(value)
    except TypeError as ex:
        raise E.InvalidArgument(f"{name} must be an integer") from ex
    if value < 0 or (positive and value == 0):
        bound = "positive" if positive else "nonnegative"
        raise E.InvalidArgument(f"{name} must be {bound}")
    return value


def _stream_copy(reader: BinaryIO, writer: BinaryIO, chunk_size: int = 1 << 20) -> int:
    """Copy binary chunks, accounting only for bytes accepted by the sink."""
    chunk_size = _stream_size(chunk_size, "chunk_size", positive=True)
    total = 0
    while True:
        chunk = reader.read(chunk_size)
        if not _is_bytes_like(chunk):
            raise E.IoError("reader.read() must return bytes-like data")
        chunk = bytes(chunk)
        if len(chunk) > chunk_size:
            raise E.IoError("reader.read() returned more bytes than requested")
        if not chunk:
            break
        offset = 0
        while offset < len(chunk):
            count = writer.write(chunk[offset:])
            if isinstance(count, bool):
                raise E.IoError("writer.write() must return an integer byte count, not bool")
            try:
                count = operator.index(count)
            except TypeError as ex:
                raise E.IoError("writer.write() must return an integer byte count", ex) from ex
            if count <= 0 or count > len(chunk) - offset:
                raise E.IoError("writer.write() returned an invalid byte count")
            offset += count
            total += count
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
        Writer SQLite cache size in kilobytes.
    reader_cache_kb : int, default=8192
        SQLite cache budget per pooled reader. Readers inherit mmap_size.
    reader_idle_limit : int, default=8
        Maximum idle readers retained, not an active-reader concurrency limit.
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
        read_only: bool = False,
        reader_cache_kb: int = 8192,
        reader_idle_limit: int = 8,
    ):
        self._chunk = _stream_size(chunk_size, "chunk_size", positive=True)
        self._retries = int(retries)
        self._backoff = float(backoff_base)
        if self._retries < 1 or not math.isfinite(self._backoff) or self._backoff < 0:
            raise E.InvalidArgument("retries must be positive and backoff_base finite/nonnegative")
        self._background_flush_error = None
        self._closed = False
        self._closing = False
        self._lifecycle_lock = threading.Lock()
        self._cache_lock_depth = 0
        self._cache_lock_transition = False
        self._cache_unlock_pending = False
        self._read_only = read_only
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
                chunk_size=self._chunk,
                enable_wal=enable_wal,
                page_size=page_size,
                mmap_size=mmap_size,
                cache_kb=cache_kb,
                read_only=read_only,
                reader_cache_kb=reader_cache_kb,
                reader_idle_limit=reader_idle_limit,
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

    def _native(self) -> _KVault:
        """Capture a live native reference without holding a lock during I/O."""
        # Capture ownership once; native admission rejects mutations after close.
        inner = self._inner
        if inner is None:
            raise ValueError("Cannot operate on closed KVault")
        return inner

    def _retry_error(self, error, method, args, key=None):
        """Allocate retry machinery only on an actual transient failure."""
        mapped = E.map_exception(error, key=key)
        if not isinstance(mapped, E.DatabaseBusy) or self._retries <= 1:
            raise mapped from error
        time.sleep(self._backoff)
        return _with_retries(
            lambda: method(*args),
            attempts=self._retries - 1,
            backoff_base=self._backoff * 2,
            key_for_error=key,
        )

    def close(self) -> None:
        """Flush and close, rejecting active cache contexts and concurrent closes.

        A failed final flush leaves the handle open and close may be retried.
        Native mutation admission fences writes against the final flush.
        """
        with self._lifecycle_lock:
            if self._closed:
                return
            if self._cache_lock_depth or self._cache_lock_transition:
                raise E.DatabaseBusy("Cannot close KVault while cache is locked")
            if self._closing:
                raise RuntimeError("KVault close is already in progress")
            self._closing = True
            inner = self._inner
            unlock_pending = self._cache_unlock_pending

        try:
            # Native callback checks run before any daemon join or stop request.
            try:
                if unlock_pending:
                    inner.set_cache_locked(False)
                    with self._lifecycle_lock:
                        self._cache_unlock_pending = False
                inner.finish_close()
            except Exception as ex:
                raise E.map_exception(ex) from ex
            with self._lifecycle_lock:
                self._closed = True
                self._inner = None
            self._stop_daemon_thread()
        finally:
            with self._lifecycle_lock:
                self._closing = False

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
            self._native().checkpoint_wal()
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

        native = self._native()
        try:
            return native.get(k)
        except Exception as ex:
            try:
                return self._retry_error(ex, native.get, (k,), k)
            except E.NotFound:
                raise KeyError(key) from None

    def __setitem__(self, key: KeyType, value: ValueType) -> None:
        """Set value for key."""
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
            return int(self._native().len())
        except Exception as ex:
            raise E.map_exception(ex)

    def __iter__(self) -> Iterator[bytes]:
        """Iterate over all keys."""
        return self.keys()

    # ----------------------------
    # Mapping Protocol
    # ----------------------------

    def _pages(self, prefix, limit, batch_size, *, items):
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")
        batch_size = operator.index(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if limit is not None:
            limit = operator.index(limit)
            if limit < 0:
                raise ValueError("limit must be nonnegative or None")
        prefix = _to_bytes_key(prefix) if prefix is not None else None
        remaining = limit
        after = None

        # Keyset pagination avoids both truncation and growing OFFSET scans.
        while remaining is None or remaining > 0:
            if self._closed:
                raise ValueError("Cannot iterate a closed KVault")
            size = batch_size if remaining is None else min(batch_size, remaining)
            scanner = self._native().scan_items if items else self._native().scan_keys
            page = _with_retries(
                lambda: scanner(prefix, size, after),
                attempts=self._retries,
                backoff_base=self._backoff,
                key_for_error=prefix,
            )
            if not page:
                break
            yield page
            after = page[-1][0] if items else page[-1]
            if remaining is not None:
                remaining -= len(page)
            if len(page) < size:
                break

    def keys(
        self,
        prefix: Optional[KeyType] = None,
        limit: int | None = None,
        *,
        batch_size: int = 1024,
    ) -> Iterator[bytes]:
        """Iterate all matching keys by default; limit is an optional total cap.

        Pages use fresh snapshots and include this handle's buffered values.
        Concurrent changes are not a single, iterator-wide snapshot.
        """
        for page in self._pages(prefix, limit, batch_size, items=False):
            yield from page

    def items(
        self,
        prefix: Optional[KeyType] = None,
        limit: int | None = None,
        *,
        batch_size: int = 128,
    ) -> Iterator[tuple[bytes, Any]]:
        """Read key/value batches without an extra query for each key."""
        for page in self._pages(prefix, limit, batch_size, items=True):
            yield from page

    def values(
        self,
        prefix: Optional[KeyType] = None,
        limit: int | None = None,
        *,
        batch_size: int = 128,
    ) -> Iterator[Any]:
        """Iterate all matching values, using bounded row batches."""
        for _, value in self.items(prefix, limit, batch_size=batch_size):
            yield value

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

        native = self._native()
        try:
            return native.get(k)
        except Exception as ex:
            try:
                return self._retry_error(ex, native.get, (k,), k)
            except E.NotFound:
                return default

    # ----------------------------
    # Core Operations
    # ----------------------------

    def put(self, key: KeyType, value: ValueType) -> None:
        """
        Store a value for a key.

        When auto-pack is enabled (default), accepts any Python object.
        When auto-pack is disabled, only accepts bytes-like objects.

        Parameters
        ----------
        key : bytes | str
            Key to store under.
        value : Any (if auto-pack enabled) | bytes (if auto-pack disabled)
            Value to store.
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")

        k = _to_bytes_key(key)

        if isinstance(value, (bytearray, memoryview)):
            value = bytes(value)
        native = self._native()
        try:
            native.put(k, value)
        except Exception as ex:
            self._retry_error(ex, native.put, (k, value), k)

    def put_many(self, items: list | tuple) -> None:
        """Commit one explicit batch atomically; duplicate keys use the last value.

        Pass a list or tuple of key/value pairs. The complete batch is serialized
        before writing, so callers control its memory bound. This bypasses the
        write-back cache and supersedes matching pending keys only on success.
        A locked enabled cache rejects the batch. ``update()`` uses bounded batches.
        """
        if not isinstance(items, (list, tuple)):
            raise E.InvalidArgument("put_many requires a list or tuple of key/value pairs")
        native = self._native()
        try:
            native.put_many(items)
        except Exception as ex:
            raise E.map_exception(ex) from ex

    def get_many(self, keys: list | tuple, default: Any = None) -> list[Any]:
        """Read an explicit key batch in input order, retaining duplicates.

        Missing keys yield ``default``; stored None remains None. Committed
        lookups share one SQLite snapshot and this handle's captured pending
        values take precedence. Concurrent cache mutations are not a combined
        cache/database transaction. Results are decoded after releasing SQLite.
        """
        if not isinstance(keys, (list, tuple)):
            raise E.InvalidArgument("get_many requires a list or tuple of keys")
        native = self._native()
        try:
            return native.get_many(keys, default)
        except Exception as ex:
            raise E.map_exception(ex) from ex

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
        ch = (
            self._chunk
            if chunk_size is None
            else _stream_size(chunk_size, "chunk_size", positive=True)
        )
        if size is not None:
            size = _stream_size(size, "size")

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
                _stream_copy(reader, buf, ch)
                return self.put(k, buf.getbuffer())
            except Exception as ex:
                raise E.map_exception(ex, key=k)

        size_int = _stream_size(inferred, "size")
        # A failed call may already have consumed input; never replay it.
        try:
            self._native().put_stream(k, reader, size_int, ch)
        except Exception as ex:
            raise E.map_exception(ex, key=k) from ex

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
        ch = (
            self._chunk
            if chunk_size is None
            else _stream_size(chunk_size, "chunk_size", positive=True)
        )

        # The sink may have accepted bytes even when the native call fails.
        try:
            return int(self._native().get_to_file(k, writer, ch))
        except Exception as ex:
            raise E.map_exception(ex, key=k) from ex

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
            return bool(self._native().delete(k))

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
            return bool(self._native().exists(k))

        return _with_retries(
            call, attempts=self._retries, backoff_base=self._backoff, key_for_error=k
        )

    def pop(self, key: KeyType, default: Any = _MISSING) -> Any:
        """
        Atomically remove and return a value; missing without default raises KeyError.

        Stored None is distinct from a missing key. SQL commits before decoding;
        a decoding error does not restore the removed raw value. No automatic retry.

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
        native = self._native()
        try:
            return native.pop(_to_bytes_key(key))
        except KeyError:
            if default is _MISSING:
                raise KeyError(key) from None
            return default
        except Exception as ex:
            raise E.map_exception(ex, key=key) from ex

    def setdefault(self, key: KeyType, default: Any = None) -> Any:
        """
        Return the decoded value, atomically inserting default (None) if absent.

        An unused default is not serialized. Insertion commits directly and is
        rejected under an enabled cache lock; existing pending values are readable.

        Parameters
        ----------
        key : bytes | str
            Key to retrieve or set.
        default : Any, default=None
            Auto-packable value to set if key doesn't exist.

        Returns
        -------
        bytes
            Existing or newly set value.
        """
        native = self._native()
        if isinstance(default, (bytearray, memoryview)):
            default = bytes(default)
        try:
            return native.setdefault(_to_bytes_key(key), default)
        except Exception as ex:
            raise E.map_exception(ex, key=key) from ex

    def update(self, other=(), **kwargs) -> None:
        """
        Update from a mapping/iterable and keyword pairs in committed batches.

        At most 1024 pairs are materialized per batch; each batch is atomic, not
        the entire update. Earlier batches remain committed on a later error.
        This bypasses cache buffering and rejects an enabled locked cache.

        Parameters
        ----------
        other : Mapping | Iterable
            Mapping or iterable of (key, value) pairs.
        **kwargs
            Additional key-value pairs.
        """
        self._native()
        pairs = chain(other.items() if isinstance(other, Mapping) else other, kwargs.items())
        while batch := list(islice(pairs, 1024)):
            self.put_many(batch)

    def clear(self) -> None:
        """
        Remove all committed and pending keys with one atomic SQL transaction.

        Failure preserves pending data. This commits directly, even inside a
        cache context, just like delete(). It never materializes every key.

        Warning: This deletes all data!
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")

        try:
            self._native().clear()
        except Exception as ex:
            raise E.map_exception(ex) from ex

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
        if flush_interval is not None and (
            not math.isfinite(flush_interval) or flush_interval <= 0
        ):
            raise E.InvalidArgument("flush_interval must be finite and positive")
        try:
            self._native().enable_cache(cap_bytes, flush_threshold, flush_interval)
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
            self._native().disable_cache()
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
            count = int(self._native().flush_cache())
            if not self._cache_lock_depth:
                self._background_flush_error = None
            return count
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

        Nested contexts defer flushing until the last context exits. This is
        not a database transaction. Close is rejected while any context is active.

        Examples
        --------
        >>> vault.enable_cache(flush_interval=5.0)
        >>> with vault.lock_cache():
        ...     vault["config:part1"] = data1
        ...     vault["config:part2"] = data2
        ... # Auto-flush deferred until here
        """
        with self._lifecycle_lock:
            if self._closed or self._closing:
                raise ValueError("Cannot lock cache on a closing or closed KVault")
            if self._cache_lock_transition:
                raise RuntimeError("Cache lock transition is in progress")
            inner = self._inner
            first = self._cache_lock_depth == 0
            if first:
                self._cache_lock_transition = True
            else:
                self._cache_lock_depth += 1

        if first:
            try:
                # Native transitions can wait; never hold the lifecycle mutex here.
                inner.set_cache_locked(True)
            except BaseException:
                with self._lifecycle_lock:
                    self._cache_lock_transition = False
                raise
            with self._lifecycle_lock:
                self._cache_lock_depth = 1
                self._cache_unlock_pending = False
                self._cache_lock_transition = False

        try:
            yield
        finally:
            with self._lifecycle_lock:
                last = self._cache_lock_depth == 1
                if last:
                    self._cache_lock_transition = True
                else:
                    self._cache_lock_depth -= 1
            if last:
                unlocked = False
                try:
                    # Retain the captured native object throughout cleanup.
                    inner.set_cache_locked(False)
                    unlocked = True
                finally:
                    with self._lifecycle_lock:
                        self._cache_lock_depth = 0
                        # A later close or context can retry a failed native unlock.
                        self._cache_unlock_pending = not unlocked
                        self._cache_lock_transition = False

    @property
    def background_flush_error(self):
        """Last daemon error, cleared by a successful unlocked manual flush.

        Failure stops that daemon and retains pending writes. Retry explicitly,
        then call enable_cache(flush_interval=...) to restart it if desired.
        """
        return self._background_flush_error

    def reader_pool_stats(self) -> dict[str, int]:
        """Return idle reader count and SQLite page-cache bytes (not total RSS)."""
        count, cache_bytes = self._native().reader_pool_stats()
        return {"idle_connections": count, "sqlite_cache_bytes": cache_bytes}

    def _start_daemon_thread(self, flush_interval: float) -> None:
        """Start background thread for auto-flushing cache after idle period."""
        # Stop existing daemon if any
        self._stop_daemon_thread()

        stop = threading.Event()

        def daemon_worker():
            last_flush = time.time()
            while not stop.wait(0.5):  # Check every 500ms
                if self._closed:
                    break

                try:
                    # Check if enough time has passed since last flush
                    if time.time() - last_flush >= flush_interval:
                        # Flush cache if not empty
                        flushed = self.flush_cache()
                        if flushed > 0:
                            last_flush = time.time()
                except Exception as ex:
                    self._background_flush_error = ex
                    try:
                        warnings.warn(
                            f"KVault background flush failed: {ex}; call flush_cache() to retry",
                            RuntimeWarning,
                            stacklevel=1,
                        )
                    finally:
                        stop.set()
                    return

        thread = threading.Thread(target=daemon_worker, daemon=True)
        with self._lifecycle_lock:
            if self._closed or self._closing:
                return
            # A timed-out older worker must never reuse this worker's stop event.
            self._daemon_stop.set()
            self._daemon_stop = stop
            self._daemon_thread = thread
            thread.start()

    def _stop_daemon_thread(self) -> None:
        """Stop the daemon without holding lifecycle state while joining it."""
        with self._lifecycle_lock:
            thread = self._daemon_thread
            self._daemon_stop.set()
        if thread is not None:
            if thread is not threading.current_thread():
                thread.join(timeout=2.0)
            with self._lifecycle_lock:
                if self._daemon_thread is thread:
                    self._daemon_thread = None

    # ----------------------------
    # Header Format Management
    # ----------------------------

    def enable_headers(self) -> None:
        """
        Enable header format for new writes.

        When enabled, values written with specific encodings will have a 10-byte
        header prepended that indicates the encoding type. This allows:
        - Auto-packing of Python objects (Phase 3)
        - Mixed encoding types in same vault
        - Future: compression/encryption flags

        IMPORTANT: Raw bytes (default put behavior) are still stored without
        headers to maintain compatibility with external tools for media files.

        Note: Existing values without headers remain readable (backward compatible).
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")
        try:
            self._native().enable_headers()
        except Exception as ex:
            raise E.map_exception(ex)

    def disable_headers(self) -> None:
        """
        Disable header format (return to raw bytes mode).

        New writes will not include headers. Existing values with headers
        can still be read.
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")
        self._native().disable_headers()

    def headers_enabled(self) -> bool:
        """
        Check if header format is currently enabled.

        Returns
        -------
        bool
            True if headers are enabled for new writes
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")
        return self._native().headers_enabled()

    # ----------------------------
    # Auto-Packing Management
    # ----------------------------

    def enable_auto_pack(self, use_pickle: bool = True) -> None:
        """
        Enable auto-packing for arbitrary Python objects (DEFAULT ENABLED).

        When enabled, KVault automatically serializes Python objects:
        - bytes → Raw (no header, media file compatible)
        - numpy arrays → DataPacker vec:*
        - int/float → DataPacker i64/f64
        - dict/list → MessagePack (efficient binary format)
        - str → DataPacker str:utf8
        - Custom objects → Pickle (if use_pickle=True)

        Values are automatically decoded on get() - you get back the original
        Python object, not bytes!

        Parameters
        ----------
        use_pickle : bool, default=True
            Allow pickle for custom objects as last resort

        Example
        -------
        >>> kv.enable_auto_pack()
        >>> kv["embedding"] = np.random.randn(768).astype(np.float32)
        >>> kv["config"] = {"timeout": 30, "retries": 3}
        >>> embedding = kv["embedding"]  # Returns np.ndarray, not bytes!
        >>> config = kv["config"]        # Returns dict, not bytes!
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")
        try:
            self._native().enable_auto_pack(use_pickle)
        except Exception as ex:
            raise E.map_exception(ex)

    def disable_auto_pack(self) -> None:
        """
        Disable auto-packing (return to bytes-only mode).

        After disabling, put() will only accept bytes and get() will
        return bytes.
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")
        self._native().disable_auto_pack()

    def auto_pack_enabled(self) -> bool:
        """
        Check if auto-packing is currently enabled.

        Returns
        -------
        bool
            True if auto-packing is enabled
        """
        if self._closed:
            raise ValueError("Cannot operate on closed KVault")
        return self._native().auto_pack_enabled()

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
            self._native().optimize()
        except Exception as ex:
            raise E.map_exception(ex)
