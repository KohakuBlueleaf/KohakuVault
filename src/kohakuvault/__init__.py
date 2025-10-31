"""
KohakuVault: SQLite-backed key-value store for large media blobs.

Features:
- Dict-like interface
- Streaming support for large files
- Write-back caching
- Thread-safe with retry logic
"""

from .proxy import KVault
from .errors import (
    KohakuVaultError,
    NotFound,
    DatabaseBusy,
    InvalidArgument,
    IoError,
)

__version__ = "0.1.0"
__all__ = [
    "KVault",
    "KohakuVaultError",
    "NotFound",
    "DatabaseBusy",
    "InvalidArgument",
    "IoError",
]
