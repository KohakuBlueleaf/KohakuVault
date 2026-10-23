"""Legacy-format, iteration, and concurrent reader regressions."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import zlib

import numpy as np
import pytest

from kohakuvault import ColumnVault, KVault, VectorKVault


@pytest.fixture
def legacy_path(tmp_path):
    source = Path(__file__).parent / "fixtures" / "legacy_0_9.db.zlib"
    path = tmp_path / "legacy.db"
    path.write_bytes(zlib.decompress(source.read_bytes()))
    return path


def _kv_bytes(path):
    with sqlite3.connect(path) as conn:
        return dict(conn.execute("SELECT key, value FROM legacy_kv"))


def test_old_kv_and_vector_storage_survives_new_columns(legacy_path):
    original = _kv_bytes(legacy_path)
    with KVault(legacy_path, table="legacy_kv") as kv:
        assert kv["raw"] == b"\x00raw\xff"
        assert kv["mapping"] == {"id": 7, "tags": ["one", "two"]}
        assert kv["integer"] == 42
        assert kv["float"] == 1.25
        assert kv["text"] == "hello"
        np.testing.assert_array_equal(kv["array"], np.arange(12, dtype=np.float32).reshape(3, 4))
        # Rewriting the same values must retain the existing byte encodings.
        for key, value in list(kv.items()):
            kv[key] = value

    with VectorKVault(legacy_path, table="legacy_vectors", dimensions=4) as vectors:
        assert vectors.count() == 2
        result = vectors.search([1, 0, 0, 0], k=1)
        assert result[0][2] == {"label": "first"}
        row_id = result[0][0]
        vectors.update(row_id, value={"label": "updated"})
        assert vectors.get_by_id(row_id)[1] == {"label": "updated"}

    with ColumnVault(legacy_path) as columns:
        columns.create_column("array", "vec:f32:4").append_array(np.eye(4, dtype=np.float32))
        np.testing.assert_array_equal(columns["array"].read_array(), np.eye(4))
    assert _kv_bytes(legacy_path) == original


def test_unlimited_paged_iteration_and_binary_prefixes(tmp_path):
    with KVault(tmp_path / "iterate.db") as kv:
        with kv.cache():
            for index in range(10_257):
                kv[f"key:{index:06d}"] = index
        assert len(list(kv)) == 10_257
        assert len(list(kv.items(batch_size=37))) == 10_257
        assert sum(kv.values(batch_size=257)) == sum(range(10_257))
        assert len(list(kv.keys(limit=1001, batch_size=100))) == 1001
        assert list(kv.keys(limit=0)) == []
        for kwargs in ({"limit": -1}, {"batch_size": 0}, {"batch_size": -1}):
            with pytest.raises(ValueError):
                list(kv.keys(**kwargs))

        keys = [b"", b"\xff", b"\xff\x00", b"\xff\xff", b"a\xff", b"a\xff\x00", b"a\xff\xff", b"b"]
        for key in keys:
            kv[key] = key
        assert list(kv.keys(prefix=b"a\xff", batch_size=1)) == keys[4:7]
        assert list(kv.keys(prefix=b"\xff", batch_size=1)) == keys[1:4]
        assert len(list(kv.keys(prefix=b""))) == len(kv)


def test_cached_iteration_overwrite_and_delete(tmp_path):
    with KVault(tmp_path / "cache.db") as kv:
        kv["a"] = b"old"
        kv.enable_cache(cap_bytes=1024, flush_threshold=1024)
        for _ in range(100):
            kv["a"] = b"replacement"
        kv["b"] = {"pending": True}
        assert len(kv) == 2
        assert "b" in kv
        assert dict(kv.items(batch_size=1)) == {b"a": b"replacement", b"b": {"pending": True}}
        del kv["b"]
        del kv["a"]
        kv.flush_cache()
        assert list(kv) == []
        assert len(kv) == 0


def test_readonly_kv_opens_under_writer_lock_and_observes_commits(tmp_path):
    path = tmp_path / "kv.db"
    with KVault(path) as writer:
        writer["value"] = 1
        conn = sqlite3.connect(path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            reader = KVault(path, read_only=True)
            assert reader["value"] == 1
        finally:
            conn.rollback()
            conn.close()
        with reader:
            writer["value"] = 2
            assert reader["value"] == 2
            with pytest.raises(Exception, match="read-only"):
                reader["value"] = 3
            assert writer["value"] == 2


def test_vector_delete_and_update_are_atomic(tmp_path):
    path = tmp_path / "vectors.db"
    with VectorKVault(path, dimensions=4) as vectors:
        row_id = vectors.insert([1, 0, 0, 0], b"original")
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TRIGGER reject_delete BEFORE DELETE ON vec_kvault_values BEGIN SELECT RAISE(ABORT, 'injected delete'); END"
            )
        with pytest.raises(Exception, match="injected delete"):
            vectors.delete(row_id)
        assert vectors.get_by_id(row_id)[1] == b"original"
        with sqlite3.connect(path) as conn:
            conn.execute("DROP TRIGGER reject_delete")
            conn.execute(
                "CREATE TRIGGER reject_update BEFORE UPDATE ON vec_kvault_values BEGIN SELECT RAISE(ABORT, 'injected update'); END"
            )
        with pytest.raises(Exception, match="injected update"):
            vectors.update(row_id, vector=[0, 1, 0, 0], value=b"new")
        vector, value = vectors.get_by_id(row_id)
        np.testing.assert_array_equal(vector, [1, 0, 0, 0])
        assert value == b"original"
        with sqlite3.connect(path) as conn:
            conn.execute("DROP TRIGGER reject_update")
        vectors.delete(row_id)
        assert vectors.count() == 0


def test_vector_readers_and_writer_share_database(tmp_path):
    path = tmp_path / "vector_concurrent.db"
    with VectorKVault(path, dimensions=4) as writer:
        writer.insert([1, 0, 0, 0], b"first")
        with VectorKVault(path, dimensions=4, read_only=True) as reader:

            def read():
                for _ in range(30):
                    assert reader.search([1, 0, 0, 0], k=1)[0][2] == b"first"

            def write():
                for index in range(30):
                    writer.insert([0, 1, 0, 0], index)

            with ThreadPoolExecutor(max_workers=4) as executor:
                jobs = [executor.submit(read) for _ in range(3)] + [executor.submit(write)]
                for job in jobs:
                    job.result(timeout=15)
            assert reader.count() == 31
            with pytest.raises(Exception, match="read-only"):
                reader.delete(1)


def test_vector_numpy_and_bytes_inputs(tmp_path):
    with VectorKVault(tmp_path / "vector_input.db", dimensions=4, metric="l2") as vectors:
        value = np.arange(8, dtype=">f8")[::2]
        row_id = vectors.insert(value, b"value")
        np.testing.assert_array_equal(vectors.get_by_id(row_id)[0], value)
        assert vectors.search(value.astype("<f4").tobytes(), k=1)[0][0] == row_id
        assert vectors.search(value, k=0) == []
        with pytest.raises(Exception):
            vectors.insert(b"bad", b"bad")
        with pytest.raises(Exception, match="metric"):
            vectors.search(value, metric="cosine")
