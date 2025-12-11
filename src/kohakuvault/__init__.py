"""
KohakuVault: SQLite-backed storage with vector search and auto-packing.

Features:
- Auto-packing: Store any Python object (numpy, dict, list, primitives)
- Vector search: Fast similarity search with sqlite-vec
- Vector storage: Efficient array/tensor storage in columns
- Streaming support for large files
- Write-back caching
- Thread-safe with retry logic
"""

__version__ = "0.8.0"

from .column_proxy import Column, ColumnVault, VarSizeColumn
from .errors import DatabaseBusy, InvalidArgument, IoError, KohakuVaultError, NotFound
from .proxy import KVault
from .text_proxy import TextVault
from .vector_proxy import VectorKVault
from .wrappers import Cbor, Json, MsgPack, Pickle

# Try to import DataPacker and CSBTree (will be available after maturin build)
try:
    from ._kvault import DataPacker

    _DATAPACKER_AVAILABLE = True
except ImportError:
    _DATAPACKER_AVAILABLE = False
    DataPacker = None

try:
    from ._kvault import CSBTree, SkipList

    _CSBTREE_AVAILABLE = True
except ImportError:
    _CSBTREE_AVAILABLE = False
    CSBTree = None
    SkipList = None

__all__ = [
    "KVault",
    "VectorKVault",
    "TextVault",
    "Column",
    "ColumnVault",
    "VarSizeColumn",
    "DataPacker",
    "CSBTree",
    "SkipList",
    "MsgPack",
    "Json",
    "Cbor",
    "Pickle",
    "KohakuVaultError",
    "NotFound",
    "DatabaseBusy",
    "InvalidArgument",
    "IoError",
    "_CSBTREE_AVAILABLE",
    "_DATAPACKER_AVAILABLE",
]
