"""Tests for VectorKVault - Vector similarity search"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from kohakuvault import VectorKVault


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


def test_vector_kvault_create(temp_db):
    """Test VectorKVault creation"""
    vkv = VectorKVault(temp_db, dimensions=128, metric="cosine")
    assert vkv.dimensions == 128
    assert vkv.metric == "cosine"
    assert vkv.vector_type == "f32"
    assert len(vkv) == 0


def test_vector_kvault_insert_and_search(temp_db):
    """Test inserting vectors and searching with numpy arrays"""
    vkv = VectorKVault(temp_db, dimensions=8, metric="cosine")

    # Insert some vectors using numpy arrays (preferred interface)
    vec1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec3 = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Similar to vec1

    id1 = vkv.insert(vec1, b"first vector")
    id2 = vkv.insert(vec2, b"second vector")
    id3 = vkv.insert(vec3, b"third vector")

    assert len(vkv) == 3
    assert vkv.exists(id1)
    assert vkv.exists(id2)
    assert vkv.exists(id3)

    # Search with query similar to vec1 (using numpy array)
    query = np.array([0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = vkv.search(query, k=2)

    assert len(results) == 2
    # Results should be sorted by distance (closest first)
    assert results[0][0] in [id1, id3]  # Should find vec1 or vec3 as closest
    assert isinstance(results[0][1], float)  # distance
    assert isinstance(results[0][2], bytes)  # value


def test_vector_kvault_get(temp_db):
    """Test get method (single closest match)"""
    vkv = VectorKVault(temp_db, dimensions=4, metric="l2")

    vec1 = [1.0, 0.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0, 0.0]

    vkv.insert(vec1, b"value1")
    vkv.insert(vec2, b"value2")

    # Query closest to vec1
    query = [0.9, 0.1, 0.0, 0.0]
    result = vkv.get(query)
    assert result == b"value1"


def test_vector_kvault_get_by_id(temp_db):
    """Test getting vector and value by ID - returns numpy array"""
    vkv = VectorKVault(temp_db, dimensions=4, metric="l2")  # Use L2 to avoid normalization

    vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    value = b"test value"

    id = vkv.insert(vec, value)

    # Get by ID - should return numpy array for vector
    retrieved_vec, retrieved_val = vkv.get_by_id(id)
    assert isinstance(retrieved_vec, np.ndarray)
    assert retrieved_vec.dtype == np.float32
    assert retrieved_vec.shape == (4,)
    np.testing.assert_array_almost_equal(retrieved_vec, vec)
    assert retrieved_val == value


def test_vector_kvault_update(temp_db):
    """Test updating vectors and values"""
    vkv = VectorKVault(temp_db, dimensions=4, metric="l2")  # Use L2 to avoid normalization

    vec = [1.0, 0.0, 0.0, 0.0]
    value = b"original"

    id = vkv.insert(vec, value)

    # Update value
    vkv.update(id, value=b"updated")
    _, new_value = vkv.get_by_id(id)
    assert new_value == b"updated"

    # Update vector
    new_vec = [0.0, 1.0, 0.0, 0.0]
    vkv.update(id, vector=new_vec)
    updated_vec, _ = vkv.get_by_id(id)
    # get_by_id now returns numpy array
    assert isinstance(updated_vec, np.ndarray)
    np.testing.assert_array_almost_equal(updated_vec, new_vec)


def test_vector_kvault_delete(temp_db):
    """Test deleting vectors"""
    vkv = VectorKVault(temp_db, dimensions=4, metric="cosine")

    vec = [1.0, 0.0, 0.0, 0.0]
    id = vkv.insert(vec, b"test")

    assert vkv.exists(id)
    assert len(vkv) == 1

    vkv.delete(id)

    assert not vkv.exists(id)
    assert len(vkv) == 0


def test_vector_kvault_metrics(temp_db):
    """Test different similarity metrics"""
    # Cosine similarity
    vkv_cosine = VectorKVault(temp_db, table="cosine_test", dimensions=4, metric="cosine")
    vec = [1.0, 0.0, 0.0, 0.0]
    vkv_cosine.insert(vec, b"test")
    results = vkv_cosine.search(vec, k=1)
    assert len(results) == 1

    # L2 (Euclidean) distance
    vkv_l2 = VectorKVault(temp_db, table="l2_test", dimensions=4, metric="l2")
    vkv_l2.insert(vec, b"test")
    results = vkv_l2.search(vec, k=1)
    assert len(results) == 1


def test_vector_kvault_info(temp_db):
    """Test info method"""
    vkv = VectorKVault(temp_db, dimensions=128, metric="cosine")

    vec = [float(i) for i in range(128)]
    vkv.insert(vec, b"test")

    info = vkv.info()
    assert info["dimensions"] == 128
    assert info["metric"] == "cosine"
    assert info["vector_type"] == "float32"
    assert info["count"] == 1


def test_vector_kvault_contains(temp_db):
    """Test __contains__ method"""
    vkv = VectorKVault(temp_db, dimensions=4, metric="cosine")

    vec = [1.0, 0.0, 0.0, 0.0]
    id = vkv.insert(vec, b"test")

    assert id in vkv
    assert 99999 not in vkv


def test_vector_kvault_repr(temp_db):
    """Test __repr__ method"""
    vkv = VectorKVault(temp_db, dimensions=128, metric="cosine")

    repr_str = repr(vkv)
    assert "VectorKVault" in repr_str
    assert "dimensions=128" in repr_str
    assert "metric='cosine'" in repr_str


def test_vector_kvault_string_values(temp_db):
    """Test storing string values (auto-converted to bytes)"""
    vkv = VectorKVault(temp_db, dimensions=4, metric="cosine")

    vec = [1.0, 0.0, 0.0, 0.0]
    id = vkv.insert(vec, "test string")

    _, value = vkv.get_by_id(id)
    assert value == b"test string"


def test_vector_kvault_dimension_validation(temp_db):
    """Test that dimension mismatch raises error"""
    vkv = VectorKVault(temp_db, dimensions=8, metric="cosine")

    # Correct dimensions
    vec_correct = [float(i) for i in range(8)]
    vkv.insert(vec_correct, b"test")

    # Wrong dimensions should fail
    vec_wrong = [1.0, 2.0, 3.0, 4.0]  # Only 4 dimensions
    with pytest.raises(Exception):  # Should raise dimension mismatch error
        vkv.insert(vec_wrong, b"test")


def test_vector_kvault_backward_compatibility_with_lists(temp_db):
    """Test that Python lists still work (backward compatibility)"""
    vkv = VectorKVault(temp_db, dimensions=4, metric="l2")

    # Should accept Python lists
    vec_list = [1.0, 2.0, 3.0, 4.0]
    id = vkv.insert(vec_list, b"list value")

    # Search should also accept lists
    query_list = [1.1, 2.1, 3.1, 4.1]
    results = vkv.search(query_list, k=1)
    assert len(results) == 1
    assert results[0][0] == id

    # get_by_id should return numpy array even when inserted as list
    vec, val = vkv.get_by_id(id)
    assert isinstance(vec, np.ndarray)
    assert val == b"list value"


def test_vector_kvault_numpy_dtypes(temp_db):
    """Test that various numpy dtypes are handled correctly"""
    vkv = VectorKVault(temp_db, dimensions=4, metric="cosine")

    # Test int array (should be converted to float)
    vec_int = np.array([1, 2, 3, 4], dtype=np.int32)
    id1 = vkv.insert(vec_int, b"int array")

    # Test float64
    vec_f64 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    id2 = vkv.insert(vec_f64, b"float64 array")

    # Test float32 (preferred)
    vec_f32 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    id3 = vkv.insert(vec_f32, b"float32 array")

    # All should be stored and retrievable
    assert vkv.exists(id1)
    assert vkv.exists(id2)
    assert vkv.exists(id3)

    # Retrieved vectors should be float32
    for id in [id1, id2, id3]:
        vec, _ = vkv.get_by_id(id)
        assert vec.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
