"""
Columnar storage for KohakuVault.

Provides list-like interface for storing large arrays/sequences in SQLite.
"""

import struct
from collections.abc import MutableSequence
from typing import Any, Iterator, Union

from kohakuvault._kvault import _ColumnVault
from kohakuvault import errors as E

# Type aliases
ValueType = Union[int, float, bytes]


# ======================================================================================
# Data Type Packers/Unpackers
# ======================================================================================


def pack_i64(value: int) -> bytes:
    """Pack int64 to 8 bytes (little-endian)."""
    return struct.pack("<q", value)


def unpack_i64(data: bytes, offset: int = 0) -> int:
    """Unpack int64 from 8 bytes (little-endian)."""
    return struct.unpack_from("<q", data, offset)[0]


def pack_f64(value: float) -> bytes:
    """Pack float64 to 8 bytes."""
    return struct.pack("<d", value)


def unpack_f64(data: bytes, offset: int = 0) -> float:
    """Unpack float64 from 8 bytes."""
    return struct.unpack_from("<d", data, offset)[0]


def pack_bytes(value: bytes, size: int) -> bytes:
    """Pack fixed-size bytes. Pads with zeros if too short."""
    if len(value) > size:
        raise ValueError(f"Value too long: {len(value)} > {size}")
    return value.ljust(size, b"\x00")


def unpack_bytes(data: bytes, offset: int, size: int) -> bytes:
    """Unpack fixed-size bytes."""
    return data[offset : offset + size]


# ======================================================================================
# Type Registry
# ======================================================================================


DTYPE_INFO = {
    "i64": {
        "elem_size": 8,
        "pack": lambda v: pack_i64(int(v)),
        "unpack": lambda d, o: unpack_i64(d, o),
    },
    "f64": {
        "elem_size": 8,
        "pack": lambda v: pack_f64(float(v)),
        "unpack": lambda d, o: unpack_f64(d, o),
    },
}


def parse_dtype(dtype: str) -> tuple[str, int]:
    """
    Parse dtype string and return (base_type, elem_size).

    Supported:
    - "i64" → ("i64", 8)
    - "f64" → ("f64", 8)
    - "bytes:N" → ("bytes", N)
    """
    if dtype in DTYPE_INFO:
        return dtype, DTYPE_INFO[dtype]["elem_size"]

    if dtype.startswith("bytes:"):
        try:
            size = int(dtype.split(":")[1])
            if size <= 0:
                raise ValueError("bytes size must be > 0")
            return "bytes", size
        except (IndexError, ValueError) as e:
            raise E.InvalidArgument(f"Invalid bytes dtype: {dtype}") from e

    raise E.InvalidArgument(f"Unknown dtype: {dtype}")


def get_packer(dtype: str, elem_size: int):
    """Get pack function for dtype."""
    if dtype in DTYPE_INFO:
        return DTYPE_INFO[dtype]["pack"]
    elif dtype == "bytes":
        return lambda v: pack_bytes(v, elem_size)
    else:
        raise E.InvalidArgument(f"No packer for dtype: {dtype}")


def get_unpacker(dtype: str, elem_size: int):
    """Get unpack function for dtype."""
    if dtype in DTYPE_INFO:
        return DTYPE_INFO[dtype]["unpack"]
    elif dtype == "bytes":
        return lambda d, o: unpack_bytes(d, o, elem_size)
    else:
        raise E.InvalidArgument(f"No unpacker for dtype: {dtype}")


# ======================================================================================
# Column Class (List-like Interface)
# ======================================================================================


class Column(MutableSequence):
    """
    List-like interface for a columnar storage.

    Supports:
    - Indexing: col[0], col[-1]
    - Assignment: col[0] = value
    - Deletion: del col[0]
    - Append: col.append(value)
    - Insert: col.insert(0, value)
    - Iteration: for x in col
    - Length: len(col)
    """

    def __init__(
        self,
        inner: _ColumnVault,
        col_id: int,
        name: str,
        dtype: str,
        elem_size: int,
        chunk_bytes: int,
    ):
        self._inner = inner
        self._col_id = col_id
        self._name = name
        self._dtype = dtype
        self._elem_size = elem_size
        self._chunk_bytes = chunk_bytes

        # Get base type for packing/unpacking
        base_dtype, _ = parse_dtype(dtype)
        self._pack = get_packer(base_dtype, elem_size)
        self._unpack = get_unpacker(base_dtype, elem_size)

        # Cache length (updated on mutations)
        self._length = None

    def _get_length(self) -> int:
        """Get current length from database."""
        if self._length is None:
            _, _, length, _ = self._inner.get_column_info(self._name)
            self._length = length
        return self._length

    def _normalize_index(self, idx: int) -> int:
        """Normalize index (handle negative indices)."""
        length = len(self)
        if idx < 0:
            idx += length
        if idx < 0 or idx >= length:
            raise IndexError(f"Column index out of range: {idx} (length={length})")
        return idx

    # ==================================================================================
    # MutableSequence Protocol
    # ==================================================================================

    def __len__(self) -> int:
        return self._get_length()

    def __getitem__(self, idx: int) -> ValueType:
        """Get element at index."""
        if not isinstance(idx, int):
            raise TypeError("Column indices must be integers")

        idx = self._normalize_index(idx)

        # Read one element
        data = self._inner.read_range(self._col_id, idx, 1, self._elem_size, self._chunk_bytes)

        return self._unpack(data, 0)

    def __setitem__(self, idx: int, value: ValueType) -> None:
        """Set element at index."""
        if not isinstance(idx, int):
            raise TypeError("Column indices must be integers")

        idx = self._normalize_index(idx)

        # Pack value
        packed = self._pack(value)

        # Write one element
        self._inner.write_range(self._col_id, idx, packed, self._elem_size, self._chunk_bytes)

    def __delitem__(self, idx: int) -> None:
        """
        Delete element at index.

        WARNING: This is O(n) - shifts all elements after idx.
        """
        if not isinstance(idx, int):
            raise TypeError("Column indices must be integers")

        idx = self._normalize_index(idx)
        length = len(self)

        if idx == length - 1:
            # Deleting last element - just update length
            self._inner.set_length(self._col_id, length - 1)
            self._length = length - 1
            return

        # Read all elements after idx
        count = length - idx - 1
        data = self._inner.read_range(
            self._col_id, idx + 1, count, self._elem_size, self._chunk_bytes
        )

        # Write them back one position earlier
        self._inner.write_range(self._col_id, idx, data, self._elem_size, self._chunk_bytes)

        # Update length
        self._inner.set_length(self._col_id, length - 1)
        self._length = length - 1

    def __iter__(self) -> Iterator[ValueType]:
        """Iterate over all elements."""
        length = len(self)
        if length == 0:
            return

        # Read in chunks for efficiency
        chunk_size = 1000
        for start in range(0, length, chunk_size):
            count = min(chunk_size, length - start)
            data = self._inner.read_range(
                self._col_id, start, count, self._elem_size, self._chunk_bytes
            )

            for i in range(count):
                yield self._unpack(data, i * self._elem_size)

    def insert(self, idx: int, value: ValueType) -> None:
        """
        Insert element at index.

        WARNING: This is O(n) - shifts all elements after idx.
        """
        length = len(self)

        # Handle negative/boundary indices
        if idx < 0:
            idx = max(0, length + idx)
        else:
            idx = min(idx, length)

        if idx == length:
            # Insert at end - just append
            self.append(value)
            return

        # Read all elements from idx to end
        count = length - idx
        data = self._inner.read_range(self._col_id, idx, count, self._elem_size, self._chunk_bytes)

        # Pack new value
        packed = self._pack(value)

        # Write new value at idx
        self._inner.write_range(self._col_id, idx, packed, self._elem_size, self._chunk_bytes)

        # Write old elements one position later
        self._inner.write_range(self._col_id, idx + 1, data, self._elem_size, self._chunk_bytes)

        # Update length
        self._inner.set_length(self._col_id, length + 1)
        self._length = length + 1

    # ==================================================================================
    # Additional Methods
    # ==================================================================================

    def append(self, value: ValueType) -> None:
        """
        Append element to end.

        This is O(1) and the most efficient operation.
        """
        packed = self._pack(value)
        current_length = self._get_length()

        self._inner.append_raw(
            self._col_id, packed, self._elem_size, self._chunk_bytes, current_length
        )

        self._length = current_length + 1

    def extend(self, values: list[ValueType]) -> None:
        """Extend column with multiple values."""
        if not values:
            return

        # Pack all values
        packed_data = b"".join(self._pack(v) for v in values)

        current_length = self._get_length()

        self._inner.append_raw(
            self._col_id, packed_data, self._elem_size, self._chunk_bytes, current_length
        )

        self._length = current_length + len(values)

    def clear(self) -> None:
        """Remove all elements."""
        self._inner.set_length(self._col_id, 0)
        self._length = 0

    def __repr__(self) -> str:
        return f"Column(name={self._name!r}, dtype={self._dtype!r}, length={len(self)})"


# ======================================================================================
# ColumnVault Class (Container)
# ======================================================================================


class ColumnVault:
    """
    Container for columnar storage.

    Usage:
        vault = ColumnVault(kvault_instance)
        vault.create_column("temperatures", "f64")
        temps = vault["temperatures"]
        temps.append(23.5)
    """

    def __init__(self, kvault_or_path: Union[Any, str], chunk_bytes: int = 1024 * 1024):
        """
        Initialize ColumnVault.

        Args:
            kvault_or_path: Either a KVault instance (to share DB) or a path string
            chunk_bytes: Default chunk size for new columns (1 MiB)
        """
        self._default_chunk_bytes = chunk_bytes

        # Get path from KVault or use string directly
        if isinstance(kvault_or_path, str):
            path = kvault_or_path
        else:
            # Assume it's a KVault instance
            path = kvault_or_path._path

        self._inner = _ColumnVault(path)
        self._columns = {}  # Cache of Column instances

    def create_column(self, name: str, dtype: str, chunk_bytes: int = None) -> "Column":
        """
        Create a new column.

        Args:
            name: Column name (must be unique)
            dtype: Data type ("i64", "f64", "bytes:N")
            chunk_bytes: Chunk size (defaults to vault default)

        Returns:
            Column instance
        """
        _, elem_size = parse_dtype(dtype)

        if chunk_bytes is None:
            chunk_bytes = self._default_chunk_bytes

        col_id = self._inner.create_column(name, dtype, elem_size, chunk_bytes)

        col = Column(self._inner, col_id, name, dtype, elem_size, chunk_bytes)
        self._columns[name] = col
        return col

    def __getitem__(self, name: str) -> "Column":
        """Get column by name."""
        if name in self._columns:
            return self._columns[name]

        # Load from database
        try:
            col_id, elem_size, length, chunk_bytes = self._inner.get_column_info(name)
        except RuntimeError as ex:
            # Convert RuntimeError from Rust to NotFound
            if "not found" in str(ex).lower():
                raise E.NotFound(name) from ex
            raise

        # Reconstruct dtype from name (stored in DB)
        # We need to get it from metadata
        cols = self._inner.list_columns()
        dtype = None
        for col_name, col_dtype, _ in cols:
            if col_name == name:
                dtype = col_dtype
                break

        if dtype is None:
            raise E.NotFound(name)

        col = Column(self._inner, col_id, name, dtype, elem_size, chunk_bytes)
        self._columns[name] = col
        return col

    def ensure(self, name: str, dtype: str, chunk_bytes: int = None) -> "Column":
        """
        Get column if exists, create if not.

        Args:
            name: Column name
            dtype: Data type (only used if creating)
            chunk_bytes: Chunk size (only used if creating)

        Returns:
            Column instance
        """
        try:
            return self[name]
        except E.NotFound:
            return self.create_column(name, dtype, chunk_bytes)

    def list_columns(self) -> list[tuple[str, str, int]]:
        """
        List all columns.

        Returns:
            List of (name, dtype, length) tuples
        """
        return self._inner.list_columns()

    def delete_column(self, name: str) -> bool:
        """
        Delete a column and all its data.

        Returns:
            True if deleted, False if not found
        """
        if name in self._columns:
            del self._columns[name]

        return self._inner.delete_column(name)

    def __repr__(self) -> str:
        cols = self.list_columns()
        return f"ColumnVault({len(cols)} columns)"
