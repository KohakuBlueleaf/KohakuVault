"""Mapping semantics, bounded updates, callback reentry, and daemon failures."""

import subprocess
import sys
import threading

import pytest

from kohakuvault import KVault
from tests.test_kv_flush_isolation import _sql


def test_pop_missing_default_and_stored_none(tmp_path):
    with KVault(tmp_path / "pop.db") as kv:
        with pytest.raises(KeyError):
            kv.pop("missing")
        marker = object()
        assert kv.pop("missing", marker) is marker
        assert kv.pop("missing", None) is None
        kv["none"] = None
        assert kv.pop("none", marker) is None
        assert "none" not in kv
        kv.enable_cache(cap_bytes=1024, flush_threshold=1024)
        kv["pending"] = {"a": 1}
        assert kv.pop("pending") == {"a": 1}
        kv.flush_cache()
        assert "pending" not in kv


def test_setdefault_preserves_types_and_existing_values(tmp_path):
    with KVault(tmp_path / "default.db") as kv:
        assert kv.setdefault("none") is None
        assert kv.setdefault("map", {"a": 1}) == {"a": 1}
        assert kv.setdefault("map", object()) == {"a": 1}
        assert kv["map"] == {"a": 1}
        kv.enable_cache(cap_bytes=1024, flush_threshold=1024)
        kv["pending"] = [1, 2]
        assert kv.setdefault("pending", "wrong") == [1, 2]


def test_clear_atomic_failure_preserves_pending(tmp_path):
    path = tmp_path / "clear.db"
    with KVault(path) as kv:
        kv.put_many([("a", b"a"), ("z", b"z")])
        kv.enable_cache(cap_bytes=1024, flush_threshold=1024)
        kv["pending"] = b"keep"
        _sql(
            path,
            "CREATE TRIGGER reject_clear BEFORE DELETE ON kv WHEN OLD.key=x'7a' BEGIN SELECT RAISE(ABORT, 'clear failure'); END;",
        )
        with pytest.raises(Exception, match="clear failure"):
            kv.clear()
        assert kv.get_many(["a", "z", "pending"]) == [b"a", b"z", b"keep"]
        _sql(path, "DROP TRIGGER reject_clear;")
        kv.clear()
        assert len(kv) == 0
        kv.flush_cache()
        assert list(kv) == []


def test_update_bounded_batches_and_kwargs(tmp_path):
    with KVault(tmp_path / "update.db") as kv:
        kv.update(((str(i), i) for i in range(2050)), extra={"yes": True})
        assert len(kv) == 2051
        assert kv["2049"] == 2049
        assert kv["extra"] == {"yes": True}
        kv.update(extra=b"overwrite")
        assert kv["extra"] == b"overwrite"


def test_background_failure_is_observable_and_retryable(tmp_path):
    path = tmp_path / "daemon.db"
    with KVault(path) as kv:
        _sql(
            path,
            "CREATE TRIGGER reject_background BEFORE INSERT ON kv BEGIN SELECT RAISE(ABORT, 'background failure'); END;",
        )
        signaled = threading.Event()

        class ObservedVault(KVault):
            def flush_cache(self):
                try:
                    return super().flush_cache()
                finally:
                    signaled.set()

        with ObservedVault(path) as writer:
            with pytest.warns(RuntimeWarning, match="background"):
                writer.enable_cache(cap_bytes=1024, flush_threshold=1024, flush_interval=0.01)
                writer["pending"] = b"keep"
                assert signaled.wait(5)
                # A daemon cycle publishes the error before its stop event is set.
                assert writer._daemon_stop.wait(5)
            assert writer.background_flush_error is not None
            assert "background failure" in str(writer.background_flush_error)
            assert writer["pending"] == b"keep"
            _sql(path, "DROP TRIGGER reject_background;")
            writer.flush_cache()
            assert writer.background_flush_error is None
            assert kv["pending"] == b"keep"


@pytest.mark.parametrize("operation", ["pop", "setdefault"])
def test_atomic_mapping_across_independent_handles(tmp_path, operation):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "atomic.db"
    with KVault(path) as first, KVault(path) as second:
        if operation == "pop":
            first["key"] = b"once"
        barrier = threading.Barrier(2)

        def call(vault, default):
            barrier.wait(timeout=5)
            return (
                vault.pop("key", None) if operation == "pop" else vault.setdefault("key", default)
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            jobs = [executor.submit(call, first, b"a"), executor.submit(call, second, b"b")]
            values = [job.result(timeout=10) for job in jobs]
        if operation == "pop":
            assert values.count(b"once") == 1
            assert values.count(None) == 1
        else:
            assert values[0] == values[1] == first["key"]


def test_update_failure_is_atomic_per_bounded_batch(tmp_path):
    path = tmp_path / "bounded.db"
    with KVault(path) as kv:

        def pairs():
            for i in range(1024):
                yield str(i), i
            yield "not-published", b"no"
            yield ("invalid",)

        with pytest.raises(Exception):
            kv.update(pairs())
        assert len(kv) == 1024
        assert kv.get("not-published") is None
        kv.enable_cache(cap_bytes=1024, flush_threshold=1024)
        with kv.lock_cache(), pytest.raises(Exception, match="locked"):
            kv.update(extra=b"forbidden")
        assert kv.get("extra") is None


@pytest.mark.parametrize("auto_pack", [False, True])
def test_mutable_buffers_are_snapshotted_as_raw_bytes(tmp_path, auto_pack):
    with KVault(tmp_path / "buffers.db") as kv:
        if not auto_pack:
            kv.disable_auto_pack()
        data = bytearray(b"original")
        kv.put_many([("array", data), ("view", memoryview(data))])
        data[:] = b"mutated!"
        assert kv.get_many(["array", "view"]) == [b"original", b"original"]


@pytest.mark.parametrize("limit", [0, 1, 8])
def test_reader_pool_memory_diagnostic(tmp_path, limit):
    with KVault(tmp_path / "pool.db", reader_idle_limit=limit, mmap_size=0) as kv:
        kv.put_many([("a", b"x" * 50000)])
        assert kv["a"] == b"x" * 50000
        stats = kv.reader_pool_stats()
        assert stats["idle_connections"] == min(limit, 1)
        assert (stats["sqlite_cache_bytes"] > 0) == (limit > 0)


def test_close_releases_database_despite_retained_exception_and_native(tmp_path):
    path = tmp_path / "release.db"
    kv = KVault(path)
    kv["present"] = b"value"
    assert kv["present"] == b"value"
    native = kv._native()
    errors = []
    try:
        kv["missing"]
    except KeyError as error:
        errors.append(error)
    assert kv.get("missing") is None
    kv.close()
    # On Windows this fails if even an idle connection survives in a traceback.
    path.unlink()
    assert errors and native is not None
    with pytest.raises(Exception, match="closed"):
        native.get(b"present")


@pytest.mark.parametrize("kind", ["kv", "vector"])
def test_serializer_and_deserializer_can_reenter_without_deadlock(tmp_path, kind):
    script = r"""
import sys
from kohakuvault import KVault, VectorKVault
kind, path = sys.argv[1:]
vault = KVault(path) if kind == "kv" else VectorKVault(path, dimensions=2)
if kind == "kv":
    vault["existing"] = b"value"
else:
    vault.insert([1, 0], b"value")
def restore():
    if kind == "kv":
        assert vault["existing"] == b"value"
        vault["decode"] = b"ok"
    else:
        assert vault.count() >= 1
        vault.auto_pack_enabled()
    return {"decoded": True}
class Callback:
    def __reduce__(self):
        if kind == "kv":
            assert vault["existing"] == b"value"
            vault["encode"] = b"ok"
        else:
            assert vault.count() >= 1
            vault.auto_pack_enabled()
        return restore, ()
if kind == "kv":
    vault["callback"] = Callback()
    assert vault["callback"] == {"decoded": True}
    vault.put_many([("batch", Callback())])
    assert vault.get_many(["batch"]) == [{"decoded": True}]
    assert vault.get_many(["encode", "decode"]) == [b"ok", b"ok"]
else:
    row = vault.insert([0, 1], Callback())
    assert vault.get_by_id(row)[1] == {"decoded": True}
vault.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, kind, str(tmp_path / "reentry.db")],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
