"""Column-specific buffer codecs; KV/vector DataPacker encodings remain unchanged."""

import math
import struct
from collections.abc import Iterable

import numpy as np

from ._kvault import DataPacker

_NUMERIC = {
    "bool": ("?", "?"),
    "i8": ("i1", "b"),
    "i16": ("<i2", "h"),
    "i32": ("<i4", "i"),
    "i64": ("<i8", "q"),
    "u8": ("u1", "B"),
    "u16": ("<u2", "H"),
    "u32": ("<u4", "I"),
    "u64": ("<u8", "Q"),
    "f16": ("<f2", "e"),
    "f32": ("<f4", "f"),
    "f64": ("<f8", "d"),
}


class ColumnCodec:
    """Encode batches without scalar boxing on numeric/tensor paths."""

    def __init__(self, dtype: str):
        self.dtype = dtype
        self.numpy_dtype = None
        self.shape = ()
        self.scalar = None
        self.packer = None
        self.encoding = None
        self.raw_bytes = dtype == "bytes"

        if dtype in _NUMERIC:
            numpy_dtype, code = _NUMERIC[dtype]
            self.numpy_dtype = np.dtype(numpy_dtype)
            self.scalar = struct.Struct("<" + code)
            self.elem_size = self.numpy_dtype.itemsize
        elif dtype.startswith(("vec:", "vector:")) and len(dtype.split(":")) > 2:
            parts = dtype.split(":")
            if parts[1] not in _NUMERIC:
                raise ValueError(f"Unsupported vector element type: {parts[1]}")
            self.numpy_dtype = np.dtype(_NUMERIC[parts[1]][0])
            self.shape = tuple(int(part) for part in parts[2:])
            if any(size <= 0 for size in self.shape):
                raise ValueError("Vector dimensions must be positive")
            self.elem_size = math.prod(self.shape) * self.numpy_dtype.itemsize
        elif dtype in ("str", "string") or (
            dtype.startswith("str:")
            and len(dtype.split(":")) == 2
            and not dtype.split(":")[1].isdigit()
        ):
            self.encoding = "utf8" if ":" not in dtype else dtype.split(":")[1]
            self.encoding = {"utf16le": "utf-16-le", "utf16be": "utf-16-be"}.get(
                self.encoding, self.encoding
            )
            "".encode(self.encoding)
            self.elem_size = 0
        elif self.raw_bytes:
            self.elem_size = 0
        else:
            # Keep generic serialized objects off the numeric fast path.
            self.packer = DataPacker(dtype)
            self.elem_size = self.packer.elem_size

        if self.elem_size > 512 << 20:
            raise ValueError("A fixed-width record cannot exceed 512 MiB")

    def pack_one(self, value) -> bytes:
        if self.scalar is not None:
            return self.scalar.pack(value)
        if self.numpy_dtype is not None:
            array = np.asarray(value, dtype=self.numpy_dtype)
            if array.shape != self.shape:
                raise ValueError(f"Expected vector shape {self.shape}, got {array.shape}")
            return array.tobytes(order="C")
        if self.raw_bytes:
            if not isinstance(value, (bytes, bytearray, memoryview)):
                raise TypeError("A bytes column requires bytes-like values")
            return bytes(value)
        if self.encoding is not None:
            if not isinstance(value, str):
                raise TypeError("A string column requires str values")
            if self.encoding == "ascii" and not value.isascii():
                raise ValueError("String contains non-ASCII characters")
            return value.encode(self.encoding)
        return self.packer.pack(value)

    def pack_batch(self, values: Iterable) -> tuple[bytes, bytes | None, int]:
        if self.numpy_dtype is not None:
            if not isinstance(values, (np.ndarray, list, tuple, memoryview)):
                values = list(values)
            try:
                array = np.asarray(values, dtype=self.numpy_dtype)
            except (TypeError, ValueError) as error:
                raise TypeError(f"Values cannot be encoded as {self.dtype}") from error
            if array.ndim == 1 and array.size == 0:
                array = array.reshape((0, *self.shape))
            if array.ndim != 1 + len(self.shape) or array.shape[1:] != self.shape:
                raise ValueError(f"Expected batch shape (N, {self.shape}), got {array.shape}")
            return array.tobytes(order="C"), None, len(array)

        items = list(values)
        if self.elem_size:
            return self.packer.pack_many(items), None, len(items)

        packed = [self.pack_one(value) for value in items]
        offsets = np.empty(len(packed) + 1, dtype="<u8")
        offsets[0] = 0
        if packed:
            np.cumsum(
                np.fromiter(map(len, packed), dtype="<u8", count=len(packed)), out=offsets[1:]
            )
        return b"".join(packed), offsets.tobytes(), len(items)

    def array(self, data: bytes, *, copy: bool = False) -> np.ndarray:
        if self.numpy_dtype is None:
            raise TypeError("Array reads require a numeric or fixed-shape vector column")
        result = np.frombuffer(data, dtype=self.numpy_dtype).reshape((-1, *self.shape))
        return result.copy() if copy else result

    def unpack_fixed(self, data: bytes) -> list:
        if self.numpy_dtype is not None:
            array = self.array(data)
            return list(array) if self.shape else array.tolist()
        return self.packer.unpack_many(data, count=len(data) // self.elem_size)

    def unpack_variable(self, data: bytes, offsets: bytes) -> list:
        positions = np.frombuffer(offsets, dtype="<u8")
        items = [data[int(a) : int(b)] for a, b in zip(positions[:-1], positions[1:])]
        if self.raw_bytes:
            return items
        if self.encoding is not None:
            return [item.decode(self.encoding) for item in items]
        return [self.packer.unpack(item, 0) for item in items]
