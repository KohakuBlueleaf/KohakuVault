"""
KohakuVault: SQLite-backed key-value store for large media blobs.

Features:
- Dict-like interface for key-value storage
- List-like interface for columnar storage
- Streaming support for large files
- Write-back caching
- Thread-safe with retry logic
"""

from .proxy import KVault
from .column_proxy import Column, ColumnVault, VarSizeColumn
from .errors import (
    KohakuVaultError,
    NotFound,
    DatabaseBusy,
    InvalidArgument,
    IoError,
)

# Try to import DataPacker (will be available after maturin build)
try:
    from ._kvault import DataPacker

    _DATAPACKER_AVAILABLE = True
except ImportError:
    _DATAPACKER_AVAILABLE = False
    DataPacker = None

__version__ = "0.4.0"
__all__ = [
    "KVault",
    "Column",
    "ColumnVault",
    "VarSizeColumn",
    "DataPacker",
    "KohakuVaultError",
    "NotFound",
    "DatabaseBusy",
    "InvalidArgument",
    "IoError",
]
