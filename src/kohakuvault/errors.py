"""
Custom exception hierarchy for KohakuVault.

These are raised by the Python proxy (KVault). The underlying Rust/PyO3 layer
may raise generic RuntimeError with string messages; we translate them here.
"""

from __future__ import annotations
from typing import Optional


class KohakuVaultError(Exception):
    """Base error for all KohakuVault exceptions."""


class NotFound(KohakuVaultError):
    """Requested key was not found."""

    def __init__(self, key: bytes | str):
        super().__init__(f"Key not found: {key!r}")
        self.key = key


class DatabaseBusy(KohakuVaultError):
    """Database is locked/busy (transient)."""

    def __init__(self, message: str = "database is busy/locked"):
        super().__init__(message)


class InvalidArgument(KohakuVaultError):
    """Invalid user input (types, ranges, etc.)."""


class IoError(KohakuVaultError):
    """I/O error while reading/writing from Python streams."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        if cause is not None:
            super().__init__(f"{message}: {cause}")
            self.__cause__ = cause
        else:
            super().__init__(message)


def _is_busy_message(msg: str) -> bool:
    """Heuristics for SQLite busy/locked messages surfaced via rusqlite/PyO3."""
    m = msg.lower()
    return (
        "database is locked" in m
        or "database table is locked" in m
        or "busy" in m
        or "locked" in m
        or "timeout" in m
        and "busy" in m
    )


def _is_not_found_message(msg: str) -> bool:
    m = msg.lower()
    return "key not found" in m or "query returned no rows" in m


def map_exception(exc: BaseException, *, key: bytes | str | None = None) -> KohakuVaultError:
    """
    Convert raw exceptions from the Rust layer or Python I/O into typed errors.
    """
    # Already typed
    if isinstance(exc, KohakuVaultError):
        return exc

    # Common PyO3 surfacing: RuntimeError with string messages
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        if _is_not_found_message(msg):
            return NotFound(key if key is not None else b"<unknown>")
        if _is_busy_message(msg):
            return DatabaseBusy(msg)
        return KohakuVaultError(msg)

    # Type/value issues from proxy validation
    if isinstance(exc, (TypeError, ValueError)):
        return InvalidArgument(str(exc))

    # I/O from Python file-like
    if isinstance(exc, OSError):
        return IoError("OS error during I/O", exc)

    # Fallback
    return KohakuVaultError(str(exc))
