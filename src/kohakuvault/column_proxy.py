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


def pack_i32(value: int) -> bytes:
    """Pack int32 to 4 bytes (little-endian)."""
    return struct.pack("<i", value)


def unpack_i32(data: bytes, offset: int = 0) -> int:
    """Unpack int32 from 4 bytes (little-endian)."""
    return struct.unpack_from("<i", data, offset)[0]


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


def parse_dtype(dtype: str) -> tuple[str, int, bool]:
    """
    Parse dtype string and return (base_type, elem_size, is_varsize).

    Supported:
    - "i64" → ("i64", 8, False)
    - "f64" → ("f64", 8, False)
    - "bytes:N" → ("bytes", N, False) - fixed-size
    - "bytes" → ("bytes", 0, True) - variable-size
    - "str:N:encoding" → ("str:N:encoding", N, False) - fixed-size string
    - "str:encoding" → ("str:encoding", 0, True) - variable-size string
    - "msgpack" → ("msgpack", 0, True) - variable-size structured
    - "cbor" → ("cbor", 0, True) - variable-size structured
    """
    # Try to use DataPacker to determine if dtype is valid
    try:
        from kohakuvault._kvault import DataPacker

        # Create packer to validate dtype and get size info
        packer = DataPacker(dtype)
        elem_size = packer.elem_size
        is_varsize = packer.is_varsize

        # Base type is the dtype itself (DataPacker handles it)
        return dtype, elem_size, is_varsize

    except (ImportError, ValueError):
        # Fallback to old behavior if DataPacker not available or dtype invalid
        pass

    # Legacy parsing (for backward compatibility)
    if dtype in DTYPE_INFO:
        return dtype, DTYPE_INFO[dtype]["elem_size"], False

    if dtype == "bytes":
        return "bytes", 0, True

    if dtype.startswith("bytes:"):
        try:
            size = int(dtype.split(":")[1])
            if size <= 0:
                raise ValueError("bytes size must be > 0")
            return "bytes", size, False
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
        chunk_bytes: int,  # This is now max_chunk_bytes from Rust
        use_rust_packer: bool = True,  # NEW: Use Rust DataPacker by default
    ):
        self._inner = inner
        self._col_id = col_id
        self._name = name
        self._dtype = dtype
        self._elem_size = elem_size
        self._chunk_bytes = chunk_bytes  # max_chunk_bytes for addressing

        # NEW: Try to use Rust DataPacker
        self._use_rust_packer = use_rust_packer
        if use_rust_packer:
            try:
                from kohakuvault._kvault import DataPacker

                self._rust_packer = DataPacker(dtype)
            except (ImportError, Exception):
                # Fall back to Python packing if DataPacker not available
                self._use_rust_packer = False

        # Keep Python packing as fallback
        if not self._use_rust_packer:
            base_dtype, _, _ = parse_dtype(dtype)
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

        # Read one element (use max_chunk_bytes for addressing)
        data = self._inner.read_range(self._col_id, idx, 1, self._elem_size, self._chunk_bytes)

        # Unpack: use Rust packer if available, otherwise Python
        if self._use_rust_packer:
            return self._rust_packer.unpack(data, 0)
        else:
            return self._unpack(data, 0)

    def __setitem__(self, idx: int, value: ValueType) -> None:
        """Set element at index."""
        if not isinstance(idx, int):
            raise TypeError("Column indices must be integers")

        idx = self._normalize_index(idx)

        # Pack value: use Rust packer if available
        if self._use_rust_packer:
            packed = self._rust_packer.pack(value)
        else:
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

            # Unpack using appropriate method
            if self._use_rust_packer:
                for i in range(count):
                    yield self._rust_packer.unpack(data, i * self._elem_size)
            else:
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
        if self._use_rust_packer:
            packed = self._rust_packer.pack(value)
        else:
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
        Now uses Rust DataPacker for ~5-10x performance improvement!
        """
        current_length = self._get_length()

        if self._use_rust_packer:
            # NEW PATH: Use Rust typed interface (packs in Rust)
            self._inner.append_typed(
                self._col_id, value, self._rust_packer, self._chunk_bytes, current_length
            )
        else:
            # OLD PATH: Python packing (fallback)
            packed = self._pack(value)
            self._inner.append_raw(
                self._col_id, packed, self._elem_size, self._chunk_bytes, current_length
            )

        self._length = current_length + 1

    def extend(self, values: list[ValueType]) -> None:
        """
        Extend column with multiple values.
        Now uses Rust DataPacker for ~10-20x performance improvement on bulk operations!
        """
        if not values:
            return

        current_length = self._get_length()

        if self._use_rust_packer:
            # NEW PATH: Use Rust typed interface (packs all in Rust with single FFI call)
            self._inner.extend_typed(
                self._col_id, values, self._rust_packer, self._chunk_bytes, current_length
            )
        else:
            # OLD PATH: Python packing (fallback)
            packed_data = b"".join(self._pack(v) for v in values)
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
# VarSizeColumn Class (Variable-Size Bytes)
# ======================================================================================


class VarSizeColumn(MutableSequence):
    """
    Variable-size bytes column using prefix sum index.

    Stores two underlying columns:
    - {name}_data: Packed bytes (all elements concatenated)
    - {name}_idx: Prefix sum of byte offsets (i64)

    This allows O(1) random access to variable-length elements.
    """

    def __init__(
        self,
        inner: _ColumnVault,
        data_col_id: int,
        idx_col_id: int,
        name: str,
        dtype: str,
        chunk_bytes: int,  # This is max_chunk_bytes from Rust
        use_rust_packer: bool = True,
    ):
        self._inner = inner
        self._data_col_id = data_col_id
        self._idx_col_id = idx_col_id
        self._name = name
        self._dtype = dtype

        # CRITICAL (v0.4.0): Get ALIGNED max_chunk_bytes for index column
        # Index has elem_size=12, so max gets aligned to (max/12)*12
        _, idx_elem_size, _, idx_max_chunk = inner.get_column_info(f"{name}_idx")
        self._idx_chunk_bytes = idx_max_chunk  # Use aligned value for index addressing

        # Data column max_chunk_bytes for adaptive append
        _, _, _, data_max_chunk = inner.get_column_info(f"{name}_data")
        self._data_max_chunk = data_max_chunk

        self._chunk_bytes = chunk_bytes  # Keep for compatibility
        self._length = None

        # NEW: Support DataPacker for structured types (msgpack, cbor, strings)
        self._use_rust_packer = use_rust_packer
        if use_rust_packer and dtype != "bytes":
            # For non-bytes variable-size types, use DataPacker
            try:
                from kohakuvault._kvault import DataPacker

                self._packer = DataPacker(dtype)
            except (ImportError, Exception):
                self._use_rust_packer = False
                self._packer = None
        else:
            self._use_rust_packer = False
            self._packer = None

    def _get_length(self) -> int:
        """Get number of elements (from index column)."""
        if self._length is None:
            _, _, length, _ = self._inner.get_column_info(f"{self._name}_idx")
            self._length = length
        return self._length

    def _get_offsets(self, start_idx: int, count: int) -> tuple[int, int]:
        """
        Get byte offsets for element range [start_idx, start_idx+count).

        Returns:
            (start_offset, end_offset) in bytes
        """
        # Read prefix sums from index column
        offsets_data = self._inner.read_range(
            self._idx_col_id, start_idx, count + 1, 8, self._chunk_bytes
        )

        # Unpack offsets
        start_offset = unpack_i64(offsets_data, 0) if start_idx > 0 else 0
        if start_idx == 0:
            start_offset = 0
            end_offset = unpack_i64(offsets_data, count * 8)
        else:
            start_offset = unpack_i64(offsets_data, 0)
            end_offset = unpack_i64(offsets_data, count * 8)

        return start_offset, end_offset

    def __len__(self) -> int:
        return self._get_length()

    def __getitem__(self, idx: int) -> bytes:
        """Get element at index using adaptive chunking (v0.4.0+)."""
        if not isinstance(idx, int):
            raise TypeError("Column indices must be integers")

        length = len(self)
        if idx < 0:
            idx += length
        if idx < 0 or idx >= length:
            raise IndexError(f"Column index out of range: {idx}")

        # NEW: Read 12-byte index (i32 chunk_id, i32 start, i32 end)
        # CRITICAL: Use aligned index chunk size, not self._chunk_bytes!
        index_data = self._inner.read_range(self._idx_col_id, idx, 1, 12, self._idx_chunk_bytes)
        chunk_id = unpack_i32(index_data, 0)
        start_byte = unpack_i32(index_data, 4)
        end_byte = unpack_i32(index_data, 8)

        # Read data from single chunk (no cross-chunk!)
        data = self._inner.read_adaptive(self._data_col_id, chunk_id, start_byte, end_byte)

        # If using packer for structured types, unpack the bytes
        if self._use_rust_packer and self._packer:
            return self._packer.unpack(bytes(data), 0)
        else:
            return bytes(data)

    def __setitem__(self, idx: int, value: bytes) -> None:
        """Set element at index (must be same size as existing element)."""
        raise NotImplementedError(
            "Setting individual elements in variable-size columns not supported. "
            "Delete and re-insert instead."
        )

    def __delitem__(self, idx: int) -> None:
        """Delete element (marks for vacuum, doesn't shift data)."""
        if not isinstance(idx, int):
            raise TypeError("Column indices must be integers")

        length = len(self)
        if idx < 0:
            idx += length
        if idx < 0 or idx >= length:
            raise IndexError(f"Column index out of range: {idx}")

        # Delete index entry (shifts remaining entries in index column)
        # This is a fixed-size column operation (12 bytes per entry)
        count = length - idx - 1
        if count > 0:
            # Read remaining index entries
            index_data = self._inner.read_range(
                self._idx_col_id, idx + 1, count, 12, self._idx_chunk_bytes
            )
            # Write them back one position earlier
            self._inner.write_range(self._idx_col_id, idx, index_data, 12, self._idx_chunk_bytes)

        # Update length
        new_length = length - 1
        self._inner.set_length(self._idx_col_id, new_length)
        self._length = new_length

        # Mark the data chunk as having deletions
        # (actual cleanup happens in vacuum)

    def __iter__(self) -> Iterator[bytes]:
        """Iterate over all elements."""
        length = len(self)
        for i in range(length):
            yield self[i]

    def insert(self, idx: int, value: bytes) -> None:
        """Insert not supported for variable-size columns."""
        raise NotImplementedError(
            "Insert not supported for variable-size columns. Use append instead."
        )

    def append(self, value) -> None:
        """Append a variable-size element using adaptive chunking (v0.4.0+)."""
        current_length = self._get_length()

        # Pack value if using DataPacker for structured types
        if self._use_rust_packer and self._packer:
            packed_value = self._packer.pack(value)
        else:
            if not isinstance(value, bytes):
                raise TypeError("Value must be bytes")
            packed_value = value

        # NEW: Use adaptive append (returns chunk_id, start, end as i32 triple)
        # Use actual data column max_chunk_bytes
        chunk_id, start_byte, end_byte = self._inner.append_raw_adaptive(
            self._data_col_id, packed_value, self._data_max_chunk
        )

        # Store 12-byte index: (i32 chunk_id, i32 start, i32 end)
        index_entry = pack_i32(chunk_id) + pack_i32(start_byte) + pack_i32(end_byte)
        # Index column has elem_size=12, use aligned chunk size for addressing
        self._inner.append_raw(
            self._idx_col_id, index_entry, 12, self._idx_chunk_bytes, current_length
        )

        self._length = current_length + 1

    def extend(self, values: list) -> None:
        """Extend with multiple variable-size elements (v0.4.0+ - optimized in Rust!)."""
        if not values:
            return

        # Pack values if using DataPacker
        if self._use_rust_packer and self._packer:
            packed_values = [self._packer.pack(v) for v in values]
        else:
            # Raw bytes - validate type
            for v in values:
                if not isinstance(v, bytes):
                    raise TypeError("Value must be bytes")
            packed_values = values

        current_length = self._get_length()

        # NEW (v0.4.0): Call extend_adaptive - ALL processing in Rust!
        # Returns packed index data (12 bytes per element)
        all_index_data = self._inner.extend_adaptive(
            self._data_col_id,
            packed_values,
            self._data_max_chunk,
        )

        # Append index data (already packed in Rust!)
        self._inner.append_raw(
            self._idx_col_id, all_index_data, 12, self._idx_chunk_bytes, current_length
        )

        self._length = current_length + len(values)

    def clear(self) -> None:
        """Remove all elements."""
        self._inner.set_length(self._data_col_id, 0)
        self._inner.set_length(self._idx_col_id, 0)
        self._length = 0

    def __repr__(self) -> str:
        return f"VarSizeColumn(name={self._name!r}, length={len(self)})"


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

    def __init__(
        self,
        kvault_or_path: Union[Any, str],
        chunk_bytes: int = 1024 * 1024,
        min_chunk_bytes: int = 128 * 1024,
        max_chunk_bytes: int = 16 * 1024 * 1024,
    ):
        """
        Initialize ColumnVault.

        Args:
            kvault_or_path: Either a KVault instance (to share DB) or a path string
            chunk_bytes: Default chunk size for new columns (1 MiB, for compatibility)
            min_chunk_bytes: Minimum chunk size (128KB, first chunk starts here)
            max_chunk_bytes: Maximum chunk size (16MB, chunks don't grow beyond this)
        """
        self._default_chunk_bytes = chunk_bytes
        self._min_chunk_bytes = min_chunk_bytes
        self._max_chunk_bytes = max_chunk_bytes

        # Get path from KVault or use string directly
        if isinstance(kvault_or_path, str):
            path = kvault_or_path
        else:
            # Assume it's a KVault instance
            path = kvault_or_path._path

        self._inner = _ColumnVault(path)
        self._columns = {}  # Cache of Column instances

    def create_column(
        self, name: str, dtype: str, chunk_bytes: int = None, use_rust_packer: bool = True
    ) -> Union["Column", "VarSizeColumn"]:
        """
        Create a new column.

        Args:
            name: Column name (must be unique)
            dtype: Data type, including:
                - Primitives: "i64", "f64", "bytes:N" (fixed-size)
                - Variable bytes: "bytes"
                - Strings: "str:utf8", "str:32:utf8", "str:ascii", "str:latin1", etc.
                - Structured: "msgpack", "cbor" (for dicts/lists)
            chunk_bytes: Chunk size (defaults to vault default)
            use_rust_packer: Use Rust DataPacker (default True, faster)

        Returns:
            Column (fixed-size) or VarSizeColumn (variable-size) instance
        """
        _, elem_size, is_varsize = parse_dtype(dtype)

        if chunk_bytes is None:
            chunk_bytes = self._default_chunk_bytes

        if is_varsize:
            # Create variable-size column (bytes, strings, msgpack, cbor)
            return self._create_varsize_column(name, dtype, chunk_bytes, use_rust_packer)

        # Create fixed-size column
        col_id = self._inner.create_column(
            name, dtype, elem_size, chunk_bytes, self._min_chunk_bytes, self._max_chunk_bytes
        )

        # IMPORTANT: Get the ALIGNED max_chunk_bytes from database (v0.4.0: element-aligned)
        _, _, _, aligned_max_chunk = self._inner.get_column_info(name)
        col = Column(
            self._inner, col_id, name, dtype, elem_size, aligned_max_chunk, use_rust_packer
        )
        self._columns[name] = col
        return col

    def _create_varsize_column(
        self, name: str, dtype: str, chunk_bytes: int, use_rust_packer: bool = True
    ) -> "VarSizeColumn":
        """Create a variable-size column using adaptive chunking (v0.4.0+)."""
        # Data column stores packed bytes (elem_size=1)
        # Store the original dtype in the data column's metadata
        data_col_id = self._inner.create_column(
            f"{name}_data", dtype, 1, chunk_bytes, self._min_chunk_bytes, self._max_chunk_bytes
        )

        # Index column stores 12-byte triples: (i32 chunk_id, i32 start, i32 end)
        idx_col_id = self._inner.create_column(
            f"{name}_idx",
            "adaptive_idx",
            12,
            chunk_bytes,
            self._min_chunk_bytes,
            self._max_chunk_bytes,
        )

        # IMPORTANT: Pass max_chunk_bytes for addressing
        col = VarSizeColumn(
            self._inner,
            data_col_id,
            idx_col_id,
            name,
            dtype,
            self._max_chunk_bytes,
            use_rust_packer,
        )
        self._columns[name] = col
        return col

    def __getitem__(self, name: str) -> Union["Column", "VarSizeColumn"]:
        """Get column by name."""
        if name in self._columns:
            return self._columns[name]

        # Check if this is a variable-size column (has _data and _idx companions)
        try:
            data_col_id, data_elem_size, data_length, data_chunk_bytes = (
                self._inner.get_column_info(f"{name}_data")
            )
            idx_col_id, idx_elem_size, idx_length, idx_chunk_bytes = self._inner.get_column_info(
                f"{name}_idx"
            )

            # Get the dtype from the _data column's metadata (we store it there)
            cols = self._inner.list_columns()
            dtype = None
            for col_name, col_dtype, _ in cols:
                if col_name == f"{name}_data":
                    dtype = col_dtype
                    break

            # This is a varsize column - use the stored dtype
            col = VarSizeColumn(
                self._inner,
                data_col_id,
                idx_col_id,
                name,
                dtype if dtype else "bytes",  # Use stored dtype from _data column
                data_chunk_bytes,
            )
            self._columns[name] = col
            return col
        except RuntimeError:
            pass  # Not a varsize column, try regular column

        # Load regular column from database
        # Get the dtype from metadata
        cols = self._inner.list_columns()
        dtype = None
        for col_name, col_dtype, _ in cols:
            if col_name == name:
                dtype = col_dtype
                break

        # Load regular column from database
        try:
            col_id, elem_size, length, chunk_bytes = self._inner.get_column_info(name)
        except RuntimeError as ex:
            # Convert RuntimeError from Rust to NotFound
            if "not found" in str(ex).lower():
                raise E.NotFound(name) from ex
            raise

        if dtype is None:
            raise E.NotFound(name)

        col = Column(self._inner, col_id, name, dtype, elem_size, chunk_bytes)
        self._columns[name] = col
        return col

    def ensure(
        self, name: str, dtype: str, chunk_bytes: int = None, use_rust_packer: bool = True
    ) -> Union["Column", "VarSizeColumn"]:
        """
        Get column if exists, create if not.

        Args:
            name: Column name
            dtype: Data type (only used if creating)
            chunk_bytes: Chunk size (only used if creating)
            use_rust_packer: Use Rust DataPacker (default True, faster)

        Returns:
            Column instance
        """
        try:
            return self[name]
        except E.NotFound:
            return self.create_column(name, dtype, chunk_bytes, use_rust_packer)

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

    def checkpoint(self) -> None:
        """
        Manually checkpoint WAL file to main database.

        This merges the WAL file into the main DB file, preventing
        the WAL from growing indefinitely. Useful for long-running
        processes with many writes.
        """
        try:
            self._inner.checkpoint_wal()
        except Exception:
            pass  # Ignore checkpoint errors (non-critical)

    def __repr__(self) -> str:
        cols = self.list_columns()
        return f"ColumnVault({len(cols)} columns)"
