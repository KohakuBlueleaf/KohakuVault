"""Explicit atomic KV batches preserve ordering, cache state and encodings."""

import sqlite3
import zlib
from pathlib import Path

import numpy as np
import pytest

from kohakuvault import KVault
from kohakuvault.errors import DatabaseBusy


@pytest.mark.parametrize("memory", [False, True])
def test_batch_order_duplicates_missing_and_none(tmp_path, memory):
    with KVault(":memory:" if memory else tmp_path / "batch.db") as kv:
        kv.put_many([(b"a", b"old"), ("b", {"v": [1, 2]}), ("a", b"new"), ("none", None)])
        marker = object()
        assert kv.get_many(["b", "missing", "a", b"a", "none"], marker) == [
            {"v": [1, 2]},
            marker,
            b"new",
            b"new",
            None,
        ]
        assert kv.get_many([]) == []
        kv.put_many(())
        assert len(kv) == 3


def test_batch_cache_precedence_and_committed_publication(tmp_path):
    path = tmp_path / "cache.db"
    with KVault(path) as kv:
        kv.put_many([("a", b"committed"), ("b", b"committed")])
        kv.enable_cache(cap_bytes=1024, flush_threshold=1024)
        kv["a"] = b"pending"
        kv["other"] = b"retain"
        assert kv.get_many(["a", "b"]) == [b"pending", b"committed"]
        kv.put_many([("a", b"batch"), ("new", b"value")])
        with KVault(path, read_only=True) as reader:
            assert reader.get_many(["a", "new", "other"]) == [b"batch", b"value", None]
            kv.flush_cache()
            assert reader.get_many(["a", "other"]) == [b"batch", b"retain"]
        with kv.lock_cache():
            with pytest.raises(DatabaseBusy):
                kv.put_many([("a", b"forbidden")])
        assert kv["a"] == b"batch"


def test_batch_failure_rolls_back_and_retains_cache(tmp_path):
    path = tmp_path / "rollback.db"
    with KVault(path) as kv:
        kv["a"] = b"old"
        conn = sqlite3.connect(path)
        try:
            conn.execute(
                "CREATE TRIGGER reject_batch BEFORE INSERT ON kv WHEN NEW.key=x'7a' BEGIN SELECT RAISE(ABORT, 'batch failure'); END"
            )
            conn.commit()
        finally:
            conn.close()
        kv.enable_cache(cap_bytes=1024, flush_threshold=1024)
        kv["a"] = b"pending"
        with pytest.raises(Exception, match="batch failure"):
            kv.put_many([("a", b"replacement"), ("new", b"partial"), ("z", b"reject")])
        assert kv.get_many(["a", "new", "z"]) == [b"pending", None, None]
        with KVault(path, read_only=True) as reader:
            assert reader.get_many(["a", "new", "z"]) == [b"old", None, None]


@pytest.mark.parametrize(
    "items",
    [
        [("a", b"partial"), (1, b"bad")],
        [("a", b"partial"), ("bad",)],
        [("a", b"partial"), object()],
    ],
)
def test_batch_validation_precedes_all_mutations(tmp_path, items):
    with KVault(tmp_path / "invalid.db") as kv:
        with pytest.raises(Exception):
            kv.put_many(items)
        assert kv.get("a") is None


def test_explicit_batches_reject_generators_without_consuming(tmp_path):
    def forbidden():
        raise AssertionError("generator was consumed")
        yield None

    with KVault(tmp_path / "iter.db") as kv:
        with pytest.raises(Exception, match="list or tuple"):
            kv.put_many(forbidden())
        with pytest.raises(Exception, match="list or tuple"):
            kv.get_many(forbidden())


def test_batch_readonly_closed_and_raw_mode(tmp_path):
    path = tmp_path / "readonly.db"
    with KVault(path) as kv:
        kv.disable_auto_pack()
        kv.put_many([("raw", b"bytes")])
        with pytest.raises(Exception):
            kv.put_many([("partial", b"no"), ("bad", {"no": True})])
        assert kv.get("partial") is None
    with KVault(path, read_only=True) as reader:
        assert reader.get_many(["raw", "missing"]) == [b"bytes", None]
        for items in ([], [("raw", b"no")]):
            with pytest.raises(Exception, match="read-only"):
                reader.put_many(items)
    for operation in (lambda: kv.get_many([]), lambda: kv.put_many([])):
        with pytest.raises(Exception, match="closed"):
            operation()


def test_bulk_rewrites_legacy_bytes_identically(tmp_path):
    path = tmp_path / "legacy.db"
    path.write_bytes(
        zlib.decompress((Path(__file__).parent / "fixtures/legacy_0_9.db.zlib").read_bytes())
    )
    conn = sqlite3.connect(path)
    try:
        original = dict(conn.execute("SELECT key, value FROM legacy_kv"))
    finally:
        conn.close()
    with KVault(path, table="legacy_kv") as kv:
        keys = list(original)
        values = kv.get_many(keys)
        kv.put_many(list(zip(keys, values)))
        np.testing.assert_array_equal(kv["array"], np.arange(12, dtype=np.float32).reshape(3, 4))
    conn = sqlite3.connect(path)
    try:
        assert dict(conn.execute("SELECT key, value FROM legacy_kv")) == original
    finally:
        conn.close()


@pytest.mark.parametrize("idle_limit", [0, 1, 8])
def test_reader_configuration_and_unicode_path(tmp_path, idle_limit):
    with KVault(
        tmp_path / "讀取.db", reader_cache_kb=128, reader_idle_limit=idle_limit, mmap_size=0
    ) as kv:
        kv.put_many([("a", 1), ("b", 2)])
        for _ in range(3):
            assert kv.get_many(["b", "a"]) == [2, 1]
    with pytest.raises(Exception, match="positive"):
        KVault(tmp_path / "invalid-reader.db", reader_cache_kb=0)


def test_uri_fallback_and_streamed_stored_headers(tmp_path):
    import io

    path = tmp_path / "uri.db"
    with KVault(path.as_uri()) as kv:
        kv.put_many([("structured", {"v": [1, 2]}), ("raw", b"bytes")])
        assert kv.get_many(["raw", "structured"]) == [b"bytes", {"v": [1, 2]}]
        sink = io.BytesIO()
        count = kv.get_to_file("structured", sink, chunk_size=3)
        conn = sqlite3.connect(path)
        try:
            stored = conn.execute("SELECT value FROM kv WHERE key=?", (b"structured",)).fetchone()[
                0
            ]
        finally:
            conn.close()
        assert sink.getvalue() == stored
        assert count == len(stored)
