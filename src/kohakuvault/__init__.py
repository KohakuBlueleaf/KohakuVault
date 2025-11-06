"""
KohakuVault: SQLite-backed key-value store for large media blobs.

Features:
- Dict-like interface for key-value storage
- List-like interface for columnar storage
- Streaming support for large files
- Write-back caching
- Thread-safe with retry logic
"""

__version__ = "0.6.0"

from .column_proxy import Column, ColumnVault, VarSizeColumn
from .errors import DatabaseBusy, InvalidArgument, IoError, KohakuVaultError, NotFound
from .proxy import KVault

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
    "Column",
    "ColumnVault",
    "VarSizeColumn",
    "DataPacker",
    "CSBTree",
    "SkipList",
    "KohakuVaultError",
    "NotFound",
    "DatabaseBusy",
    "InvalidArgument",
    "IoError",
    "_CSBTREE_AVAILABLE",
    "_DATAPACKER_AVAILABLE",
]
