"""
TextVault - Full-text search storage with BM25 ranking

This module provides a KVault-like interface with full-text search capabilities.
Built on top of SQLite's FTS5 extension.

Features:
- Full-text search with BM25 ranking
- Exact match key-value lookups
- Flexible multi-column schema
- Auto-packing for arbitrary Python values
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ._kvault import _TextVault
import kohakuvault.errors as E

TextInput = Union[str, Dict[str, str]]


class TextVault:
    """Full-text search with arbitrary values using FTS5 BM25 ranking.

    Provides both exact key-value lookups and full-text search interfaces.
    Supports flexible multi-column schemas for structured document storage.

    Args:
        path: Database file path
        table: Table name (default: "text_vault")
        columns: List of indexed text columns (default: ["content"])

    Examples:
        >>> from kohakuvault import TextVault
        >>>
        >>> # Single column (simple text search)
        >>> tv = TextVault("data.db")
        >>> doc_id = tv.insert("Hello world, this is a test", {"metadata": "value"})
        >>> results = tv.search("hello", k=10)
        >>> for id, score, value in results:
        ...     print(f"ID {id}: score={score:.4f}")
        >>>
        >>> # Multi-column (structured documents)
        >>> tv = TextVault("data.db", columns=["title", "body", "tags"])
        >>> doc_id = tv.insert(
        ...     {"title": "Introduction", "body": "This is the content...", "tags": "intro tutorial"},
        ...     {"author": "John", "date": "2024-01-01"}
        ... )
        >>> results = tv.search("title:Introduction", k=10)
        >>>
        >>> # Exact key-value lookup (single column only)
        >>> tv = TextVault("data.db")
        >>> tv["unique key"] = {"data": "value"}
        >>> retrieved = tv["unique key"]
        >>>
        >>> # Search with snippets (highlighted matches)
        >>> results = tv.search_with_snippets("hello world", k=10)
        >>> for id, score, snippet, value in results:
        ...     print(f"Snippet: {snippet}")
    """

    def __init__(
        self,
        path: str,
        table: str = "text_vault",
        columns: Optional[List[str]] = None,
    ):
        self._vault = _TextVault(path, table, columns)
        self.path = path
        self.table = table
        self._columns = columns or ["content"]

    @property
    def columns(self) -> List[str]:
        """Get the list of indexed text columns."""
        return self._vault.columns()

    def insert(
        self,
        texts: TextInput,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a document with text content and value.

        Args:
            texts: Text content - string for single column, dict for multiple columns
            value: Value to store (any Python object when auto-pack enabled, bytes otherwise)
            metadata: Optional dict of metadata (not yet supported)

        Returns:
            ID of inserted document
        """
        try:
            return self._vault.insert(texts, value, metadata)
        except Exception as ex:
            raise E.map_exception(ex)

    def search(
        self,
        query: str,
        k: int = 10,
        column: Optional[str] = None,
        escape: bool = True,
    ) -> List[Tuple[int, float, Any]]:
        """Search documents using FTS5 with BM25 ranking.

        Args:
            query: Search query string
            k: Maximum number of results to return (default: 10)
            column: Optional column to search in (for multi-column vaults)
            escape: If True (default), escape special FTS5 characters for safe
                    literal matching. Handles ?, +, @, etc. automatically.
                    Set to False to use raw FTS5 query syntax (AND, OR, NOT, *, etc.)

        Returns:
            List of (id, bm25_score, value) tuples, sorted by relevance
            Higher scores indicate more relevant results
            Values are auto-decoded to original Python types

        Examples:
            # Safe literal search (default, escape=True)
            tv.search("What is this?")  # Works with special chars
            tv.search("C++ programming")
            tv.search("test@email.com")

            # Raw FTS5 syntax (escape=False)
            tv.search("hello AND world", escape=False)
            tv.search("python OR java", escape=False)
            tv.search("hello*", escape=False)  # Prefix match
        """
        try:
            return self._vault.search(query, k, column, escape)
        except Exception as ex:
            raise E.map_exception(ex)

    def search_with_text(
        self,
        query: str,
        k: int = 10,
        column: Optional[str] = None,
        escape: bool = True,
    ) -> List[Tuple[int, float, TextInput, Any]]:
        """Search and return documents with their text content.

        Args:
            query: Search query string
            k: Maximum number of results
            column: Optional column to search in
            escape: If True (default), escape special FTS5 characters

        Returns:
            List of (id, bm25_score, texts, value) tuples
            texts is string (single column) or dict (multiple columns)
        """
        try:
            return self._vault.search_with_text(query, k, column, escape)
        except Exception as ex:
            raise E.map_exception(ex)

    def search_with_snippets(
        self,
        query: str,
        k: int = 10,
        snippet_column: Optional[str] = None,
        snippet_tokens: int = 10,
        highlight_start: str = "**",
        highlight_end: str = "**",
        escape: bool = True,
    ) -> List[Tuple[int, float, str, Any]]:
        """Search with highlighted snippets.

        Returns search results with matching text snippets highlighted.

        Args:
            query: Search query string
            k: Maximum number of results
            snippet_column: Column to generate snippet from (default: first column)
            snippet_tokens: Number of tokens around match (default: 10)
            highlight_start: Highlight start marker (default: "**")
            highlight_end: Highlight end marker (default: "**")
            escape: If True (default), escape special FTS5 characters

        Returns:
            List of (id, bm25_score, snippet, value) tuples
        """
        try:
            return self._vault.search_with_snippets(
                query, k, snippet_column, snippet_tokens, highlight_start, highlight_end, escape
            )
        except Exception as ex:
            raise E.map_exception(ex)

    def get(self, key: str) -> Any:
        """Get value by exact key match (single-column TextVault only).

        Uses FTS5 exact phrase matching.

        Args:
            key: Exact text to match

        Returns:
            Value associated with the key (auto-decoded to original Python type)

        Raises:
            KeyError: If key not found
            ValueError: If vault has multiple columns
        """
        try:
            return self._vault.get(key)
        except Exception as ex:
            raise E.map_exception(ex, key=key)

    def get_by_id(self, id: int) -> Tuple[TextInput, Any]:
        """Get document text and value by ID.

        Args:
            id: Row ID

        Returns:
            (texts, value) tuple where:
            - texts is string (single column) or dict (multiple columns)
            - value is auto-decoded to original Python type

        Raises:
            NotFound: If ID doesn't exist
        """
        try:
            return self._vault.get_by_id(id)
        except Exception as ex:
            raise E.map_exception(ex)

    def delete(self, id: int) -> None:
        """Delete document by ID.

        Args:
            id: Row ID to delete
        """
        try:
            self._vault.delete(id)
        except Exception as ex:
            raise E.map_exception(ex)

    def update(
        self,
        id: int,
        texts: Optional[TextInput] = None,
        value: Optional[Any] = None,
    ) -> None:
        """Update document text or value by ID.

        Args:
            id: Row ID
            texts: New text content (optional)
            value: New value (any Python object when auto-pack enabled) (optional)

        Raises:
            ValueError: If neither texts nor value provided
        """
        try:
            self._vault.update(id, texts, value)
        except Exception as ex:
            raise E.map_exception(ex)

    def exists(self, id: int) -> bool:
        """Check if ID exists.

        Args:
            id: Row ID

        Returns:
            True if exists
        """
        try:
            return self._vault.exists(id)
        except Exception as ex:
            raise E.map_exception(ex)

    def count(self) -> int:
        """Get total count of documents.

        Returns:
            Number of documents in database
        """
        try:
            return self._vault.count()
        except Exception as ex:
            raise E.map_exception(ex)

    def count_matches(self, query: str) -> int:
        """Count documents matching a query.

        Args:
            query: FTS5 query string

        Returns:
            Number of matching documents
        """
        try:
            return self._vault.count_matches(query)
        except Exception as ex:
            raise E.map_exception(ex)

    def info(self) -> Dict[str, Any]:
        """Get TextVault info.

        Returns:
            Dict with table info (table, columns, count)
        """
        try:
            return self._vault.info()
        except Exception as ex:
            raise E.map_exception(ex)

    def clear(self) -> None:
        """Clear all documents from the vault."""
        try:
            self._vault.clear()
        except Exception as ex:
            raise E.map_exception(ex)

    def keys(self, limit: int = 10000, offset: int = 0) -> List[int]:
        """Get document IDs with pagination.

        Args:
            limit: Maximum number of IDs to return (default: 10000)
            offset: Number of IDs to skip (default: 0)

        Returns:
            List of rowids in the vault
        """
        try:
            return self._vault.keys(limit, offset)
        except Exception as ex:
            raise E.map_exception(ex)

    # ----------------------------
    # Auto-packing Management (EXACTLY SAME AS KVault/VectorKVault)
    # ----------------------------

    def enable_auto_pack(self, use_pickle: bool = True) -> None:
        """
        Enable auto-packing for arbitrary Python objects (DEFAULT ENABLED).

        When enabled, TextVault automatically serializes Python objects:
        - bytes → Raw (no header, media file compatible)
        - numpy arrays → DataPacker vec:*
        - int/float → DataPacker i64/f64
        - dict/list → MessagePack (efficient binary format)
        - str → DataPacker str:utf8
        - Custom objects → Pickle (if use_pickle=True)

        Values are automatically decoded on get() - you get back the original
        Python object, not bytes!

        Parameters
        ----------
        use_pickle : bool, default=True
            Allow pickle for custom objects as last resort
        """
        try:
            self._vault.enable_auto_pack(use_pickle)
        except Exception as ex:
            raise E.map_exception(ex)

    def disable_auto_pack(self) -> None:
        """
        Disable auto-packing (return to bytes-only mode).

        After disabling, insert() will only accept bytes and get() will
        return bytes.
        """
        try:
            self._vault.disable_auto_pack()
        except Exception as ex:
            raise E.map_exception(ex)

    def auto_pack_enabled(self) -> bool:
        """
        Check if auto-packing is currently enabled.

        Returns
        -------
        bool
            True if auto-packing is enabled
        """
        try:
            return self._vault.auto_pack_enabled()
        except Exception as ex:
            raise E.map_exception(ex)

    def enable_headers(self) -> None:
        """Enable header format for new writes."""
        try:
            self._vault.enable_headers()
        except Exception as ex:
            raise E.map_exception(ex)

    def disable_headers(self) -> None:
        """Disable header format (return to raw bytes mode)."""
        try:
            self._vault.disable_headers()
        except Exception as ex:
            raise E.map_exception(ex)

    def headers_enabled(self) -> bool:
        """Check if headers are enabled."""
        try:
            return self._vault.headers_enabled()
        except Exception as ex:
            raise E.map_exception(ex)

    # ----------------------------
    # Dict-like Interface (key-value style for single-column vaults)
    # ----------------------------

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Insert text-value pair using dict-like syntax.

        tv["key text"] = value

        This is a convenience method for single-column vaults.
        Note: The ID is not returned, so this is mainly for write-heavy workloads.

        Args:
            key: Text content (key)
            value: Value to store (any Python object when auto-pack enabled)
        """
        self.insert(key, value)

    def __getitem__(self, key: str) -> Any:
        """
        Get value by exact key match using dict-like syntax.

        value = tv["key text"]

        This is equivalent to get(key) - returns the value for exact match.
        Only works for single-column vaults.

        Args:
            key: Exact text to match

        Returns:
            Value associated with the key (auto-decoded to original type)

        Raises:
            KeyError: If key not found
        """
        try:
            return self.get(key)
        except (E.NotFound, E.KohakuVaultError) as ex:
            raise KeyError(key) from ex

    def __delitem__(self, key: str) -> None:
        """
        Delete by exact key match using dict-like syntax.

        del tv["key text"]

        Only works for single-column vaults. Finds the exact match and deletes it.

        Args:
            key: Exact text to match

        Raises:
            KeyError: If key not found
        """
        # Find the document with exact match
        try:
            results = self._vault.search(f'"{key}"', 1, None)
            if not results:
                raise KeyError(key)
            doc_id = results[0][0]
            self.delete(doc_id)
        except Exception as ex:
            if isinstance(ex, KeyError):
                raise
            raise E.map_exception(ex, key=key)

    def __len__(self) -> int:
        """Return count of documents."""
        return self.count()

    def __contains__(self, id: int) -> bool:
        """Check if ID exists."""
        return self.exists(id)

    def __repr__(self) -> str:
        return (
            f"TextVault(path={self.path!r}, table={self.table!r}, "
            f"columns={self.columns!r}, count={len(self)})"
        )
