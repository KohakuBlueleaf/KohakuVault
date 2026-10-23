"""Buffer-oriented columns backed by transactional SQLite blocks."""

import operator
import os
import struct
import threading
import warnings
from collections.abc import Iterator, MutableSequence
from contextlib import contextmanager

import numpy as np

from . import errors as E
from ._kvault import _ColumnVault
from .column_codec import ColumnCodec


def parse_dtype(dtype: str) -> tuple[str, int, bool]:
    codec = ColumnCodec(dtype)
    return dtype, codec.elem_size, codec.elem_size == 0


class _ColumnState:
    """Shared lifetime state without a vault/column reference cycle."""

    def __init__(self, read_only: bool):
        self.closed = False
        self.closing = False
        self.read_only = read_only
        self.error = None

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError("ColumnVault is closed")
        if self.error is not None:
            error, self.error = self.error, None
            raise RuntimeError("Automatic column flush failed; pending data is retained") from error

    def _check_writable(self) -> None:
        self._check_open()
        if self.closing:
            raise ValueError("ColumnVault is closing")
        if self.read_only:
            raise PermissionError("ColumnVault is read-only")


class Column(MutableSequence):
    """A typed column with scalar convenience and contiguous batch operations."""

    def __init__(self, vault, col_id: int, name: str, codec: ColumnCodec, block_bytes: int):
        self._vault = vault._state
        self._inner = vault._inner
        self._col_id = col_id
        self._name = name
        self._codec = codec
        self._dtype = codec.dtype
        self._elem_size = codec.elem_size
        self._block_bytes = block_bytes
        self._lock = threading.RLock()
        self._cache_enabled = False
        self._cap_bytes = 0
        self._flush_threshold = 0
        self._buffer = bytearray()
        self._offsets = bytearray(8)
        self._pending_rows = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> str:
        return self._dtype

    def __len__(self) -> int:
        self._vault._check_open()
        with self._lock:
            return self._inner.length(self._col_id) + self._pending_rows

    def _index(self, index: int) -> int:
        try:
            index = operator.index(index)
        except TypeError as error:
            raise TypeError("Column indices must be integers or slices") from error
        length = len(self)
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError("Column index out of range")
        return index

    def _read_ready(self) -> None:
        self._vault._check_open()
        # Reads on a buffering writer publish its pending rows first.
        if self._cache_enabled:
            self.flush_cache()

    def read_buffer(self, start: int = 0, count: int | None = None) -> bytes:
        """Read packed fixed-width bytes without constructing scalar objects."""
        self._read_ready()
        if not self._elem_size:
            raise TypeError("Use read_buffers() for variable-width columns")
        return self._inner.read_fixed(self._col_id, start, count)

    def read_buffers(self, start: int = 0, count: int | None = None) -> tuple[bytes, np.ndarray]:
        """Read variable-width payload and u64 offsets in one snapshot."""
        self._read_ready()
        if self._elem_size:
            raise TypeError("Use read_buffer() for fixed-width columns")
        payload, offsets = self._inner.read_variable(self._col_id, start, count)
        return payload, np.frombuffer(offsets, dtype="<u8")

    def read_array(
        self, start: int = 0, count: int | None = None, *, copy: bool = False
    ) -> np.ndarray:
        """Return a read-only numeric batch, or a writable copy when requested."""
        return self._codec.array(self.read_buffer(start, count), copy=copy)

    def read(self, start: int = 0, count: int | None = None) -> list:
        self._read_ready()
        if self._elem_size:
            return self._codec.unpack_fixed(self._inner.read_fixed(self._col_id, start, count))
        payload, offsets = self._inner.read_variable(self._col_id, start, count)
        return self._codec.unpack_variable(payload, offsets)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step == 1:
                return self.read(start, max(0, stop - start))
            raise ValueError("Step slicing not supported; use take(indices) instead")
        return self.read(self._index(key), 1)[0]

    def take(self, indices, *, as_array: bool = True):
        """Gather fixed-width rows by block, preserving order and duplicates."""
        self._read_ready()
        indices = [operator.index(index) for index in indices]
        if not self._elem_size:
            # Variable-width gathers remain a convenience path.
            return [self[index] for index in indices]
        data = self._inner.take_fixed(self._col_id, indices)
        if as_array and self._codec.numpy_dtype is not None:
            return self._codec.array(data)
        return self._codec.unpack_fixed(data)

    def iter_batches(
        self,
        batch_size: int | None = None,
        *,
        batch_bytes: int = 1 << 20,
        start: int = 0,
        stop: int | None = None,
        as_array: bool = True,
    ) -> Iterator:
        """Iterate to the initial row count, with no total-row limit."""
        self._read_ready()
        if batch_bytes <= 0:
            raise ValueError("batch_bytes must be positive")
        if batch_size is None:
            batch_size = max(1, batch_bytes // (self._elem_size or 256))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        length = len(self)
        start, stop, _ = slice(start, stop).indices(length)

        for offset in range(start, stop, batch_size):
            count = min(batch_size, stop - offset)
            if as_array and self._codec.numpy_dtype is not None:
                yield self.read_array(offset, count)
            else:
                yield self.read(offset, count)

    def __iter__(self) -> Iterator:
        for batch in self.iter_batches(as_array=False):
            yield from batch

    def _commit(self, payload: bytes, offsets: bytes | None) -> None:
        if self._elem_size:
            self._inner.append_fixed(self._col_id, payload)
        else:
            self._inner.append_variable(self._col_id, payload, offsets)

    def _buffer_bytes(self) -> int:
        return len(self._buffer) + (len(self._offsets) - 8 if not self._elem_size else 0)

    def _queue(self, payload: bytes, offsets: bytes | None, count: int) -> None:
        # Recheck after serialization and acquiring the column lock.
        self._vault._check_writable()
        if count == 0:
            return
        cost = len(payload) + (count * 8 if offsets is not None else 0)
        if not self._cache_enabled:
            self._commit(payload, offsets)
            return

        if self._buffer_bytes() + cost > self._cap_bytes:
            self.flush_cache()
        if cost > self._cap_bytes:
            # Oversized batches bypass the cache instead of breaking its bound.
            self._commit(payload, offsets)
            return

        base = len(self._buffer)
        self._buffer.extend(payload)
        if offsets is not None:
            if count == 1:
                self._offsets.extend(struct.pack("<Q", len(self._buffer)))
            else:
                shifted = np.frombuffer(offsets, dtype="<u8")[1:] + base
                self._offsets.extend(shifted.astype("<u8", copy=False).tobytes())
        self._pending_rows += count
        if self._buffer_bytes() >= self._flush_threshold:
            self.flush_cache()

    def append(self, value) -> None:
        self._vault._check_writable()
        payload = self._codec.pack_one(value)
        offsets = struct.pack("<QQ", 0, len(payload)) if not self._elem_size else None
        with self._lock:
            self._queue(payload, offsets, 1)

    def extend(self, values) -> None:
        self._vault._check_writable()
        payload, offsets, count = self._codec.pack_batch(values)
        with self._lock:
            self._queue(payload, offsets, count)

    def append_array(self, values: np.ndarray) -> None:
        """Append an (N, *shape) numeric batch without per-element serialization."""
        if self._codec.numpy_dtype is None:
            raise TypeError("append_array requires a numeric or fixed-shape vector column")
        self.extend(values)

    def __setitem__(self, key, values) -> None:
        self._vault._check_writable()
        with self._lock:
            self.flush_cache()
            if isinstance(key, slice):
                start, stop, step = key.indices(len(self))
                values = list(values)
                if len(values) != len(range(start, stop, step)):
                    raise ValueError(
                        "Slice length mismatch: assignment must preserve the number of rows"
                    )
                if step != 1:
                    raise ValueError("Step slicing not supported; use explicit updates instead")
            else:
                start = self._index(key)
                stop = start + 1
                values = [values]

            if self._elem_size:
                payload, _, _ = self._codec.pack_batch(values)
                self._inner.overwrite_fixed(self._col_id, start, payload)
            else:
                current = self.read()
                current[start:stop] = values
                self._replace(current)

    def _replace(self, values) -> None:
        payload, offsets, _ = self._codec.pack_batch(values)
        if self._elem_size:
            self._inner.replace_fixed(self._col_id, payload)
        else:
            self._inner.replace_variable(self._col_id, payload, offsets)

    def __delitem__(self, key) -> None:
        self._vault._check_writable()
        with self._lock:
            self.flush_cache()
            current = self.read()
            del current[key]
            self._replace(current)

    def insert(self, index: int, value) -> None:
        self._vault._check_writable()
        with self._lock:
            self.flush_cache()
            current = self.read()
            current.insert(operator.index(index), value)
            self._replace(current)

    def clear(self) -> None:
        self._vault._check_writable()
        with self._lock:
            self.flush_cache()
            self._inner.clear(self._col_id)

    def enable_cache(self, cap_bytes: int = 8 << 20, flush_threshold: int | None = None) -> None:
        self._vault._check_writable()
        if cap_bytes <= 0 or (flush_threshold is not None and flush_threshold <= 0):
            raise ValueError("Cache sizes must be positive")
        with self._lock:
            self.flush_cache()
            self._cap_bytes = cap_bytes
            self._flush_threshold = min(cap_bytes, flush_threshold or (1 << 20))
            self._cache_enabled = True

    def flush_cache(self) -> int:
        self._vault._check_open()
        with self._lock:
            if not self._pending_rows:
                return 0
            size = self._buffer_bytes()
            offsets = bytes(self._offsets) if not self._elem_size else None
            self._commit(bytes(self._buffer), offsets)

            # A failed transaction leaves the complete buffer available for retry.
            self._buffer.clear()
            self._offsets = bytearray(8)
            self._pending_rows = 0
            return size

    def disable_cache(self) -> None:
        with self._lock:
            self.flush_cache()
            self._cache_enabled = False

    @contextmanager
    def cache(self, cap_bytes: int = 8 << 20, flush_threshold: int | None = None):
        previous = (self._cap_bytes, self._flush_threshold) if self._cache_enabled else None
        self.enable_cache(cap_bytes, flush_threshold)
        try:
            yield self
        finally:
            if previous is None:
                self.disable_cache()
            else:
                self.enable_cache(*previous)

    def storage_info(self) -> dict[str, int]:
        self._read_ready()
        blocks, used, capacity = self._inner.storage_info(self._col_id)
        return {"blocks": blocks, "payload_bytes": used, "capacity_bytes": capacity}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, dtype={self.dtype!r}, length={len(self)})"
        )


class VarSizeColumn(Column):
    """Variable-width column using bounded payload blocks and u64 offsets."""


class ColumnVault:
    """Column catalog sharing a SQLite file with KV and vector vaults.

    The new column format does not read legacy col_* tables. Use close() or a
    context manager to flush pending writes and release file handles explicitly.
    """

    def __init__(
        self,
        kvault_or_path,
        min_chunk_bytes: int | None = None,
        max_chunk_bytes: int | None = None,
        *,
        block_bytes: int = 1 << 20,
        read_only: bool = False,
    ):
        # Old sizing keywords remain API aliases, not legacy-format support.
        if max_chunk_bytes is not None:
            block_bytes = max_chunk_bytes
        if block_bytes <= 0 or block_bytes > 512 << 20:
            raise ValueError("block_bytes must be between 1 and 512 MiB")
        if min_chunk_bytes is not None and not 0 < min_chunk_bytes <= block_bytes:
            raise ValueError("min_chunk_bytes must be positive and no larger than block_bytes")

        source = getattr(kvault_or_path, "_path", kvault_or_path)
        self._path = os.fspath(source)
        self._inner = _ColumnVault(self._path, read_only)
        self._block_bytes = block_bytes
        self._read_only = read_only
        self._state = _ColumnState(read_only)
        self._columns = {}
        self._lock = threading.RLock()
        self._cache_config = None
        self._pause_flush = 0
        self._daemon_stop = threading.Event()
        self._daemon_thread = None

    @property
    def _closed(self) -> bool:
        return self._state.closed

    def _check_open(self) -> None:
        self._state._check_open()

    def _check_writable(self) -> None:
        self._state._check_writable()

    def create_column(
        self,
        name: str,
        dtype: str,
        chunk_bytes: int | None = None,
        use_rust_packer: bool = True,
        *,
        block_bytes: int | None = None,
    ) -> Column:
        self._check_writable()
        codec = ColumnCodec(dtype)
        size = block_bytes if block_bytes is not None else chunk_bytes
        size = self._block_bytes if size is None else size
        with self._lock:
            self._check_writable()
            col_id = self._inner.create(name, dtype, codec.elem_size, size)
            cls = Column if codec.elem_size else VarSizeColumn
            column = cls(self, col_id, name, codec, size)
            self._columns[name] = column
            if self._cache_config is not None:
                column.enable_cache(*self._cache_config)
            return column

    def __getitem__(self, name: str) -> Column:
        self._check_open()
        with self._lock:
            try:
                col_id, dtype, _, _, block_bytes = self._inner.info(name)
            except KeyError as error:
                raise E.NotFound(name) from error
            cached = self._columns.get(name)
            if cached is not None and cached._col_id == col_id:
                return cached
            codec = ColumnCodec(dtype)
            cls = Column if codec.elem_size else VarSizeColumn
            column = cls(self, col_id, name, codec, block_bytes)
            self._columns[name] = column
            if self._cache_config is not None:
                column.enable_cache(*self._cache_config)
            return column

    def ensure(self, name: str, dtype: str, **options) -> Column:
        try:
            column = self[name]
        except E.NotFound:
            return self.create_column(name, dtype, **options)
        if column.dtype != dtype:
            raise ValueError(f"Column {name!r} has dtype {column.dtype!r}, not {dtype!r}")
        return column

    def list_columns(self) -> list[tuple[str, str, int]]:
        self._check_open()
        return self._inner.list_columns()

    def delete_column(self, name: str) -> bool:
        self._check_writable()
        with self._lock:
            column = self._columns.get(name)
            if column is not None:
                column.flush_cache()
            deleted = self._inner.delete_column(name)
            self._columns.pop(name, None)
            return deleted

    def checkpoint(self) -> tuple[int, int, int]:
        self._check_writable()
        self.flush_cache()
        return self._inner.checkpoint_wal()

    def enable_cache(
        self,
        cap_bytes: int = 8 << 20,
        flush_threshold: int | None = None,
        flush_interval: float | None = None,
    ) -> None:
        self._check_writable()
        if cap_bytes <= 0 or (flush_threshold is not None and flush_threshold <= 0):
            raise ValueError("Cache sizes must be positive")
        if flush_interval is not None and flush_interval <= 0:
            raise ValueError("flush_interval must be positive")
        with self._lock:
            for column in self._columns.values():
                column.enable_cache(cap_bytes, flush_threshold)
            self._cache_config = (cap_bytes, flush_threshold)
        if flush_interval is not None and self._daemon_thread is None:
            self._daemon_stop.clear()
            self._daemon_thread = threading.Thread(
                target=self._flush_worker,
                args=(flush_interval,),
                daemon=True,
            )
            self._daemon_thread.start()

    def _stop_worker(self) -> None:
        self._daemon_stop.set()
        worker = self._daemon_thread
        if worker is not None:
            worker.join()
            self._daemon_thread = None

    def _flush_worker(self, interval: float) -> None:
        while not self._daemon_stop.wait(interval):
            with self._lock:
                paused = self._pause_flush > 0
            if not paused:
                try:
                    self.flush_cache()
                except Exception as error:
                    self._state.error = error
                    return

    def flush_cache(self) -> int:
        self._check_open()
        with self._lock:
            columns = list(self._columns.values())
        return sum(column.flush_cache() for column in columns)

    def disable_cache(self) -> None:
        self._stop_worker()
        self._check_open()
        with self._lock:
            for column in self._columns.values():
                column.disable_cache()
            self._cache_config = None

    @contextmanager
    def cache(self, cap_bytes: int = 8 << 20, flush_threshold: int | None = None):
        previous = self._cache_config
        self.enable_cache(cap_bytes, flush_threshold)
        try:
            yield self
        finally:
            if previous is None:
                self.disable_cache()
            else:
                self.enable_cache(*previous)

    @contextmanager
    def lock_cache(self):
        """Pause timer flushes; this is not a multi-column transaction."""
        with self._lock:
            self._pause_flush += 1
        try:
            yield self
        finally:
            with self._lock:
                self._pause_flush -= 1

    def close(self) -> None:
        if self._closed:
            return
        self._state.closing = True
        try:
            self._stop_worker()
            self.flush_cache()
            self._inner.close()
            self._state.closed = True
        finally:
            self._state.closing = False

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        if hasattr(self, "_closed") and not self._closed:
            try:
                self.close()
            except Exception:
                warnings.warn("ColumnVault closed without a successful flush", ResourceWarning)

    def __repr__(self) -> str:
        return f"ColumnVault({self._path!r}, closed={self._closed})"
