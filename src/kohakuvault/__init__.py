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

__version__ = "0.2.1"
__all__ = [
    "KVault",
    "Column",
    "ColumnVault",
    "VarSizeColumn",
    "KohakuVaultError",
    "NotFound",
    "DatabaseBusy",
    "InvalidArgument",
    "IoError",
]
