"""Tests for TextVault - Full-text search with FTS5 BM25 ranking"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from kohakuvault import TextVault


@pytest.fixture
def temp_db():
    """Create a temporary database file"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    Path(db_path).unlink(missing_ok=True)
    Path(f"{db_path}-shm").unlink(missing_ok=True)
    Path(f"{db_path}-wal").unlink(missing_ok=True)


# ============================================
# Basic Creation and Info Tests
# ============================================


def test_text_vault_create_default(temp_db):
    """Test TextVault creation with default settings"""
    tv = TextVault(temp_db)
    assert tv.columns == ["content"]
    assert len(tv) == 0


def test_text_vault_create_single_column(temp_db):
    """Test TextVault creation with single column"""
    tv = TextVault(temp_db, columns=["text"])
    assert tv.columns == ["text"]
    assert len(tv) == 0


def test_text_vault_create_multi_column(temp_db):
    """Test TextVault creation with multiple columns"""
    tv = TextVault(temp_db, columns=["title", "body", "tags"])
    assert tv.columns == ["title", "body", "tags"]
    assert len(tv) == 0


def test_text_vault_info(temp_db):
    """Test info method"""
    tv = TextVault(temp_db, columns=["title", "content"])
    tv.insert({"title": "Test", "content": "Hello"}, b"value")

    info = tv.info()
    assert info["table"] == "text_vault"
    assert info["columns"] == ["title", "content"]
    assert info["count"] == 1


def test_text_vault_repr(temp_db):
    """Test __repr__ method"""
    tv = TextVault(temp_db, columns=["content"])
    repr_str = repr(tv)
    assert "TextVault" in repr_str
    assert "content" in repr_str


# ============================================
# Insert and Basic CRUD Tests
# ============================================


def test_text_vault_insert_single_column_string(temp_db):
    """Test inserting with single column using string"""
    tv = TextVault(temp_db, columns=["content"])

    doc_id = tv.insert("Hello world, this is a test document", b"value1")

    assert doc_id >= 1
    assert len(tv) == 1
    assert tv.exists(doc_id)


def test_text_vault_insert_single_column_dict(temp_db):
    """Test inserting with single column using dict"""
    tv = TextVault(temp_db, columns=["content"])

    doc_id = tv.insert({"content": "Hello world"}, b"value1")

    assert doc_id >= 1
    assert len(tv) == 1


def test_text_vault_insert_multi_column(temp_db):
    """Test inserting with multiple columns"""
    tv = TextVault(temp_db, columns=["title", "body", "tags"])

    doc_id = tv.insert(
        {"title": "Introduction", "body": "This is the main content", "tags": "intro tutorial"},
        b"metadata",
    )

    assert doc_id >= 1
    assert len(tv) == 1


def test_text_vault_get_by_id_single_column(temp_db):
    """Test getting document by ID for single column vault"""
    tv = TextVault(temp_db, columns=["content"])

    doc_id = tv.insert("Hello world", b"test value")
    texts, value = tv.get_by_id(doc_id)

    assert texts == "Hello world"
    assert value == b"test value"


def test_text_vault_get_by_id_multi_column(temp_db):
    """Test getting document by ID for multi column vault"""
    tv = TextVault(temp_db, columns=["title", "body"])

    doc_id = tv.insert({"title": "Test Title", "body": "Test body content"}, b"value")
    texts, value = tv.get_by_id(doc_id)

    assert isinstance(texts, dict)
    assert texts["title"] == "Test Title"
    assert texts["body"] == "Test body content"
    assert value == b"value"


def test_text_vault_update_text(temp_db):
    """Test updating document text"""
    tv = TextVault(temp_db, columns=["content"])

    doc_id = tv.insert("Original text", b"value")
    tv.update(doc_id, texts="Updated text")

    texts, _ = tv.get_by_id(doc_id)
    assert texts == "Updated text"


def test_text_vault_update_value(temp_db):
    """Test updating document value"""
    tv = TextVault(temp_db, columns=["content"])

    doc_id = tv.insert("Some text", b"original")
    tv.update(doc_id, value=b"updated")

    _, value = tv.get_by_id(doc_id)
    assert value == b"updated"


def test_text_vault_delete(temp_db):
    """Test deleting document"""
    tv = TextVault(temp_db, columns=["content"])

    doc_id = tv.insert("Text to delete", b"value")
    assert tv.exists(doc_id)
    assert len(tv) == 1

    tv.delete(doc_id)

    assert not tv.exists(doc_id)
    assert len(tv) == 0


def test_text_vault_clear(temp_db):
    """Test clearing all documents"""
    tv = TextVault(temp_db, columns=["content"])

    tv.insert("First", b"1")
    tv.insert("Second", b"2")
    tv.insert("Third", b"3")
    assert len(tv) == 3

    tv.clear()

    assert len(tv) == 0


def test_text_vault_keys(temp_db):
    """Test getting all document IDs"""
    tv = TextVault(temp_db, columns=["content"])

    id1 = tv.insert("First", b"1")
    id2 = tv.insert("Second", b"2")
    id3 = tv.insert("Third", b"3")

    keys = tv.keys()
    assert len(keys) == 3
    assert id1 in keys
    assert id2 in keys
    assert id3 in keys


def test_text_vault_keys_pagination(temp_db):
    """Test keys() with limit and offset"""
    tv = TextVault(temp_db, columns=["content"])

    # Insert 10 documents
    ids = [tv.insert(f"Doc {i}", b"value") for i in range(10)]

    # Get first 3
    keys = tv.keys(limit=3)
    assert len(keys) == 3

    # Get next 3
    keys = tv.keys(limit=3, offset=3)
    assert len(keys) == 3

    # Get all with high limit
    keys = tv.keys(limit=100)
    assert len(keys) == 10


# ============================================
# FTS5 Search Tests (BM25 Ranking)
# ============================================


def test_text_vault_search_basic(temp_db):
    """Test basic full-text search"""
    tv = TextVault(temp_db, columns=["content"])

    tv.insert("The quick brown fox jumps over the lazy dog", b"doc1")
    tv.insert("Python is a great programming language", b"doc2")
    tv.insert("The fox is quick and clever", b"doc3")

    results = tv.search("fox", k=10)

    assert len(results) == 2
    # Results should be (id, score, value) tuples
    for doc_id, score, value in results:
        assert isinstance(doc_id, int)
        assert isinstance(score, float)
        assert score > 0  # Positive scores (we convert from negative BM25)


def test_text_vault_search_phrase(temp_db):
    """Test phrase search"""
    tv = TextVault(temp_db, columns=["content"])

    tv.insert("The quick brown fox", b"doc1")
    tv.insert("quick fox", b"doc2")
    tv.insert("brown fox", b"doc3")

    # Exact phrase search
    results = tv.search('"quick brown"', k=10)

    assert len(results) == 1
    assert results[0][2] == b"doc1"


def test_text_vault_search_and_operator(temp_db):
    """Test AND operator in search"""
    tv = TextVault(temp_db, columns=["content"])

    tv.insert("apple banana cherry", b"doc1")
    tv.insert("apple orange", b"doc2")
    tv.insert("banana grape", b"doc3")

    # AND search (need escape=False for raw FTS5 syntax)
    results = tv.search("apple AND banana", k=10, escape=False)

    assert len(results) == 1
    assert results[0][2] == b"doc1"


def test_text_vault_search_or_operator(temp_db):
    """Test OR operator in search"""
    tv = TextVault(temp_db, columns=["content"])

    tv.insert("apple fruit", b"doc1")
    tv.insert("orange fruit", b"doc2")
    tv.insert("grape fruit", b"doc3")

    # OR search (need escape=False for raw FTS5 syntax)
    results = tv.search("apple OR orange", k=10, escape=False)

    assert len(results) == 2


def test_text_vault_search_not_operator(temp_db):
    """Test NOT operator in search"""
    tv = TextVault(temp_db, columns=["content"])

    tv.insert("python programming", b"doc1")
    tv.insert("python snake", b"doc2")
    tv.insert("java programming", b"doc3")

    # NOT search (need escape=False for raw FTS5 syntax)
    results = tv.search("python NOT snake", k=10, escape=False)

    assert len(results) == 1
    assert results[0][2] == b"doc1"


def test_text_vault_search_column_prefix(temp_db):
    """Test column prefix search for multi-column vaults"""
    tv = TextVault(temp_db, columns=["title", "body"])

    tv.insert({"title": "Python Guide", "body": "Learn programming basics"}, b"doc1")
    tv.insert({"title": "Java Guide", "body": "Python comparison included"}, b"doc2")
    tv.insert({"title": "Ruby Tutorial", "body": "Another programming language"}, b"doc3")

    # Search only in title column (need escape=False for column prefix syntax)
    results = tv.search("title:Python", k=10, escape=False)

    assert len(results) == 1
    assert results[0][2] == b"doc1"


def test_text_vault_search_column_parameter(temp_db):
    """Test column parameter for search"""
    tv = TextVault(temp_db, columns=["title", "body"])

    tv.insert({"title": "Python Guide", "body": "Learn programming basics"}, b"doc1")
    tv.insert({"title": "Java Guide", "body": "Python comparison included"}, b"doc2")

    # Search only in body column using parameter
    results = tv.search("Python", k=10, column="body")

    assert len(results) == 1
    assert results[0][2] == b"doc2"


def test_text_vault_search_ranking(temp_db):
    """Test that results are ranked by relevance"""
    tv = TextVault(temp_db, columns=["content"])

    # Doc with more occurrences should rank higher
    tv.insert("python python python programming", b"doc1")
    tv.insert("python programming", b"doc2")
    tv.insert("javascript programming", b"doc3")

    results = tv.search("python", k=10)

    assert len(results) == 2
    # First result should have higher score (more python mentions)
    assert results[0][1] >= results[1][1]


def test_text_vault_search_limit(temp_db):
    """Test search result limit"""
    tv = TextVault(temp_db, columns=["content"])

    for i in range(20):
        tv.insert(f"Document number {i} with keyword", b"value")

    results = tv.search("keyword", k=5)

    assert len(results) == 5


def test_text_vault_search_with_text(temp_db):
    """Test search_with_text returns document content"""
    tv = TextVault(temp_db, columns=["title", "body"])

    tv.insert({"title": "Test Title", "body": "Test body content"}, b"value")

    results = tv.search_with_text("Test", k=10)

    assert len(results) == 1
    doc_id, score, texts, value = results[0]
    assert isinstance(texts, dict)
    assert texts["title"] == "Test Title"
    assert texts["body"] == "Test body content"


def test_text_vault_search_with_snippets(temp_db):
    """Test search with highlighted snippets"""
    tv = TextVault(temp_db, columns=["content"])

    tv.insert(
        "This is a long document about Python programming. Python is great for data science.",
        b"value",
    )

    results = tv.search_with_snippets("Python", k=10, highlight_start="<b>", highlight_end="</b>")

    assert len(results) == 1
    doc_id, score, snippet, value = results[0]
    assert "<b>Python</b>" in snippet


def test_text_vault_count_matches(temp_db):
    """Test counting matching documents"""
    tv = TextVault(temp_db, columns=["content"])

    tv.insert("apple banana", b"1")
    tv.insert("apple orange", b"2")
    tv.insert("banana grape", b"3")
    tv.insert("cherry melon", b"4")

    assert tv.count_matches("apple") == 2
    assert tv.count_matches("banana") == 2
    assert tv.count_matches("cherry") == 1
    assert tv.count_matches("pineapple") == 0


# ============================================
# Exact Match (Key-Value) Tests
# ============================================


def test_text_vault_exact_match_get(temp_db):
    """Test exact key-value get"""
    tv = TextVault(temp_db, columns=["content"])

    tv.insert("unique_key_123", b"my_value")
    tv.insert("another_key", b"other_value")

    result = tv.get("unique_key_123")
    assert result == b"my_value"


def test_text_vault_exact_match_not_found(temp_db):
    """Test exact match raises KeyError when not found"""
    tv = TextVault(temp_db, columns=["content"])

    tv.insert("existing_key", b"value")

    with pytest.raises(Exception):  # Should be KeyError
        tv.get("nonexistent_key")


def test_text_vault_dict_interface_set_get(temp_db):
    """Test dict-like set and get"""
    tv = TextVault(temp_db, columns=["content"])

    tv["my_key"] = b"my_value"

    result = tv["my_key"]
    assert result == b"my_value"


def test_text_vault_dict_interface_keyerror(temp_db):
    """Test dict-like access raises KeyError"""
    tv = TextVault(temp_db, columns=["content"])

    with pytest.raises(KeyError):
        _ = tv["nonexistent"]


def test_text_vault_dict_interface_delete(temp_db):
    """Test dict-like delete"""
    tv = TextVault(temp_db, columns=["content"])

    tv["my_key"] = b"value"
    assert len(tv) == 1

    del tv["my_key"]
    assert len(tv) == 0


def test_text_vault_contains(temp_db):
    """Test __contains__ (checks ID existence)"""
    tv = TextVault(temp_db, columns=["content"])

    doc_id = tv.insert("test", b"value")

    assert doc_id in tv
    assert 99999 not in tv


# ============================================
# Auto-packing Tests
# ============================================


def test_text_vault_auto_pack_enabled_by_default(temp_db):
    """Test that auto-pack is enabled by default"""
    tv = TextVault(temp_db)
    assert tv.auto_pack_enabled() is True
    assert tv.headers_enabled() is True


def test_text_vault_auto_pack_dict(temp_db):
    """Test auto-packing dicts"""
    tv = TextVault(temp_db)

    test_dict = {"name": "test", "value": 123, "nested": {"key": "value"}}
    doc_id = tv.insert("test document", test_dict)

    _, retrieved = tv.get_by_id(doc_id)
    assert isinstance(retrieved, dict)
    assert retrieved == test_dict


def test_text_vault_auto_pack_list(temp_db):
    """Test auto-packing lists"""
    tv = TextVault(temp_db)

    test_list = [1, 2, 3, "four", 5.0]
    doc_id = tv.insert("test document", test_list)

    _, retrieved = tv.get_by_id(doc_id)
    assert isinstance(retrieved, list)
    assert retrieved == test_list


def test_text_vault_auto_pack_numpy(temp_db):
    """Test auto-packing numpy arrays"""
    tv = TextVault(temp_db)

    test_array = np.array([10, 20, 30], dtype=np.int64)
    doc_id = tv.insert("test document", test_array)

    _, retrieved = tv.get_by_id(doc_id)
    assert isinstance(retrieved, np.ndarray)
    np.testing.assert_array_equal(retrieved, test_array)


def test_text_vault_auto_pack_primitives(temp_db):
    """Test auto-packing int/float/str"""
    tv = TextVault(temp_db)

    id1 = tv.insert("doc1", 42)
    id2 = tv.insert("doc2", 3.14159)
    id3 = tv.insert("doc3", "hello world")

    _, val1 = tv.get_by_id(id1)
    assert isinstance(val1, int)
    assert val1 == 42

    _, val2 = tv.get_by_id(id2)
    assert isinstance(val2, float)
    assert abs(val2 - 3.14159) < 1e-10

    _, val3 = tv.get_by_id(id3)
    assert isinstance(val3, str)
    assert val3 == "hello world"


def test_text_vault_auto_pack_bytes_stays_raw(temp_db):
    """Test that bytes stay raw"""
    tv = TextVault(temp_db)

    test_bytes = b"\xff\xd8\xff\xe0jpeg data"
    doc_id = tv.insert("test document", test_bytes)

    _, retrieved = tv.get_by_id(doc_id)
    assert isinstance(retrieved, bytes)
    assert retrieved == test_bytes


def test_text_vault_search_returns_decoded_values(temp_db):
    """Test that search() returns auto-decoded values"""
    tv = TextVault(temp_db)

    tv.insert("dict document", {"type": "dict"})
    tv.insert("list document", [1, 2, 3])
    tv.insert("string document", "string value")

    results = tv.search("document", k=10)

    assert len(results) == 3
    for doc_id, score, value in results:
        assert not isinstance(value, bytes)
        assert isinstance(value, (dict, list, str))


def test_text_vault_enable_disable_auto_pack(temp_db):
    """Test enable_auto_pack() and disable_auto_pack()"""
    tv = TextVault(temp_db)

    assert tv.auto_pack_enabled() is True

    tv.disable_auto_pack()
    assert tv.auto_pack_enabled() is False

    tv.enable_auto_pack()
    assert tv.auto_pack_enabled() is True


def test_text_vault_mixed_types_same_vault(temp_db):
    """Test storing different types in same vault"""
    tv = TextVault(temp_db)

    test_data = [
        ("doc1", {"type": "dict"}, dict),
        ("doc2", [1, 2, 3], list),
        ("doc3", np.array([10, 20, 30], dtype=np.int64), np.ndarray),
        ("doc4", 42, int),
        ("doc5", 3.14, float),
        ("doc6", "string", str),
        ("doc7", b"bytes", bytes),
    ]

    ids = []
    for text, value, expected_type in test_data:
        doc_id = tv.insert(text, value)
        ids.append((doc_id, expected_type))

    for doc_id, expected_type in ids:
        _, retrieved = tv.get_by_id(doc_id)
        assert isinstance(retrieved, expected_type)


# ============================================
# Edge Cases and Error Handling
# ============================================


def test_text_vault_empty_search(temp_db):
    """Test search on empty vault"""
    tv = TextVault(temp_db)

    results = tv.search("anything", k=10)
    assert len(results) == 0


def test_text_vault_special_characters_in_text(temp_db):
    """Test handling special characters in text"""
    tv = TextVault(temp_db)

    tv.insert("Test with 'quotes' and \"double quotes\"", b"value")
    tv.insert("Test with unicode: 日本語 한국어 中文", b"value2")

    assert len(tv) == 2


def test_text_vault_search_special_characters(temp_db):
    """Test searching for text with special FTS5 characters"""
    tv = TextVault(temp_db)

    tv.insert("What is this?", b"doc1")
    tv.insert("C++ programming guide", b"doc2")
    tv.insert("Contact: test@email.com", b"doc3")
    tv.insert("Price > 100", b"doc4")
    tv.insert('He said "hello" to me', b"doc5")

    # All these should work with escape=True (default)
    assert len(tv.search("What is this?")) == 1
    assert len(tv.search("C++")) == 1
    assert len(tv.search("test@email.com")) == 1
    assert len(tv.search("Price > 100")) == 1

    # Search for text containing quotes
    results = tv.search("hello")
    assert len(results) == 1
    assert results[0][2] == b"doc5"


def test_text_vault_long_text(temp_db):
    """Test handling long text documents"""
    tv = TextVault(temp_db)

    long_text = "word " * 10000  # 10000 words
    doc_id = tv.insert(long_text, b"value")

    texts, _ = tv.get_by_id(doc_id)
    assert len(texts) == len(long_text)


def test_text_vault_invalid_column_name(temp_db):
    """Test that invalid column names are rejected"""
    with pytest.raises(Exception):
        TextVault(temp_db, columns=["valid", "invalid column"])

    with pytest.raises(Exception):
        TextVault(temp_db, columns=["valid", ""])


def test_text_vault_multi_column_with_string_raises(temp_db):
    """Test that multi-column vault rejects plain string insert"""
    tv = TextVault(temp_db, columns=["title", "body"])

    with pytest.raises(Exception):
        tv.insert("plain string", b"value")  # Should require dict


def test_text_vault_get_on_multi_column_raises(temp_db):
    """Test that get() on multi-column vault raises ValueError"""
    tv = TextVault(temp_db, columns=["title", "body"])
    tv.insert({"title": "test", "body": "content"}, b"value")

    with pytest.raises(Exception):  # Should be ValueError
        tv.get("test")


def test_text_vault_nonexistent_id(temp_db):
    """Test accessing nonexistent ID"""
    tv = TextVault(temp_db)

    with pytest.raises(Exception):
        tv.get_by_id(99999)


# ============================================
# RAG-Specific Use Cases
# ============================================


def test_text_vault_rag_document_chunks(temp_db):
    """Test storing and searching document chunks for RAG"""
    tv = TextVault(temp_db, columns=["chunk"])

    # Simulate chunked documents
    chunks = [
        ("Machine learning is a subset of artificial intelligence", {"source": "doc1", "chunk": 0}),
        ("Deep learning uses neural networks with many layers", {"source": "doc1", "chunk": 1}),
        (
            "Natural language processing enables computers to understand text",
            {"source": "doc2", "chunk": 0},
        ),
        ("Transformers are the foundation of modern NLP models", {"source": "doc2", "chunk": 1}),
    ]

    for text, metadata in chunks:
        tv.insert(text, metadata)

    # Search for relevant chunks
    results = tv.search("neural networks", k=2)

    assert len(results) > 0
    # Should find the deep learning chunk
    found_deep_learning = any(
        r[2].get("source") == "doc1" and r[2].get("chunk") == 1 for r in results
    )
    assert found_deep_learning


def test_text_vault_rag_with_embeddings_metadata(temp_db):
    """Test storing chunks with embedding vectors as metadata"""
    tv = TextVault(temp_db, columns=["content"])

    # Store text chunk with its embedding as metadata
    chunk = "This is a sample chunk for RAG"
    embedding = np.random.randn(384).astype(np.float32)  # Simulated embedding
    metadata = {"embedding": embedding, "doc_id": "doc1"}

    doc_id = tv.insert(chunk, metadata)

    # Retrieve and verify
    _, retrieved_meta = tv.get_by_id(doc_id)
    assert retrieved_meta["doc_id"] == "doc1"
    np.testing.assert_array_almost_equal(retrieved_meta["embedding"], embedding)


def test_text_vault_rag_hybrid_search(temp_db):
    """Test text search results can be used for hybrid search with vector"""
    tv = TextVault(temp_db, columns=["content"])

    # Store documents with embeddings
    docs = [
        ("Python programming basics", np.random.randn(128).astype(np.float32)),
        ("Advanced Python techniques", np.random.randn(128).astype(np.float32)),
        ("JavaScript fundamentals", np.random.randn(128).astype(np.float32)),
    ]

    ids = []
    for text, embedding in docs:
        doc_id = tv.insert(text, {"text": text, "embedding": embedding})
        ids.append(doc_id)

    # BM25 search first
    bm25_results = tv.search("Python", k=10)

    # Results contain embeddings for potential re-ranking
    assert len(bm25_results) == 2
    for doc_id, score, value in bm25_results:
        assert "embedding" in value
        assert isinstance(value["embedding"], np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
