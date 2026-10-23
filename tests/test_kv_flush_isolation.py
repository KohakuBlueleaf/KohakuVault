"""Deterministic pending/in-flight KV isolation and ordering regressions.

Each scenario runs in a fresh process so a native lock regression cannot hang
pytest. The SQLite writer reservation is held by a *different* process: Python
and Rust may load SQLite libraries with independent process-lock registries.
"""

import io
import subprocess
import sys
import threading
import time

import pytest

from kohakuvault import KVault
from kohakuvault.errors import DatabaseBusy
from tests.test_cache_consistency import _sqlite_writer_lock

_DEADLINE = 20


class _Worker:
    def __init__(self, operation):
        self.started = threading.Event()
        self.done = threading.Event()
        self.value = None
        self.error = None

        def run():
            self.started.set()
            try:
                self.value = operation()
            except BaseException as error:
                self.error = error
            finally:
                self.done.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        assert self.started.wait(_DEADLINE), "worker did not start"

    def wait(self):
        assert self.done.wait(_DEADLINE), "worker did not finish"
        self.thread.join(_DEADLINE)
        assert not self.thread.is_alive()
        return self

    def result(self):
        self.wait()
        if self.error is not None:
            raise self.error
        return self.value


def _inflight(vault, entries):
    # Observe the native state transition, not merely a thread entering Python.
    deadline = time.monotonic() + _DEADLINE
    pause = threading.Event()
    while True:
        state = vault._native().cache_stats()
        if state[1] == entries:
            return state
        assert time.monotonic() < deadline, f"batch never became in-flight: {state}"
        pause.wait(0.01)


def _run(scenario, path, *arguments):
    result = subprocess.run(
        [sys.executable, "-m", "tests.test_kv_flush_isolation", scenario, str(path), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _sql(path, statement):
    # Do not mix Python sqlite3 and the native writer's process-scoped locks.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sqlite3, sys; c = sqlite3.connect(sys.argv[1]); "
            "c.executescript(sys.argv[2]); c.close()",
            str(path),
            statement,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=_DEADLINE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _enable(vault, capacity=1 << 20):
    vault.enable_cache(cap_bytes=capacity, flush_threshold=capacity, flush_interval=None)


def _committed(path, expected):
    with KVault(path, read_only=True) as observer:
        for key, value in expected.items():
            assert observer.get(key) == value
            assert (key in observer) == (value is not None)


def test_blocked_flush_allows_committed_inflight_and_newer_pending_reads(tmp_path):
    _run("visibility", tmp_path / "visibility.db")


def _visibility(path):
    with KVault(path, retries=1) as vault:
        vault["committed"] = b"persisted"
        vault["changing"] = b"persisted-old"
        _enable(vault)
        vault["changing"] = b"inflight-old"
        vault["inflight-only"] = b"batch-only"
        with _sqlite_writer_lock(path):
            flush = _Worker(vault.flush_cache)
            assert _inflight(vault, 2)[:2] == (0, 2)
            assert _Worker(lambda: vault["committed"]).result() == b"persisted"
            assert _Worker(lambda: vault["inflight-only"]).result() == b"batch-only"
            assert _Worker(lambda: vault["changing"]).result() == b"inflight-old"
            _Worker(lambda: vault.put("changing", b"newer-pending")).result()
            assert _Worker(lambda: vault["changing"]).result() == b"newer-pending"
            assert vault._native().cache_stats()[:2] == (1, 2)
            assert vault.get_many(["changing", "inflight-only", "committed"]) == [
                b"newer-pending",
                b"batch-only",
                b"persisted",
            ]
            assert len(vault) == 3
            assert dict(vault.items(batch_size=1)) == {
                b"changing": b"newer-pending",
                b"inflight-only": b"batch-only",
                b"committed": b"persisted",
            }
            _committed(path, {"changing": b"persisted-old", "inflight-only": None})
            assert not flush.done.is_set(), "flush escaped the external writer reservation"
        assert flush.result() == 2
        assert vault["changing"] == b"newer-pending"
        _committed(path, {"changing": b"inflight-old", "inflight-only": b"batch-only"})
        assert vault.flush_cache() == 1
        _committed(path, {"changing": b"newer-pending"})
        assert vault._native().cache_stats() == (0, 0, 0)


def test_capacity_includes_inflight_and_reports_backpressure_without_data_loss(tmp_path):
    _run("capacity", tmp_path / "capacity.db")


def _capacity(path):
    with KVault(path, retries=1) as vault:
        _enable(vault, 64)
        vault[b"a"] = b"a" * 40
        assert vault._native().cache_stats() == (1, 0, 41)
        with _sqlite_writer_lock(path):
            flush = _Worker(vault.flush_cache)
            assert _inflight(vault, 1) == (0, 1, 41)
            _Worker(lambda: vault.put(b"b", b"b" * 20)).result()
            assert vault._native().cache_stats() == (1, 1, 62)
            pressure = _Worker(lambda: vault.put(b"c", b"c" * 10))
            # Keep the reservation until SQLite returns its normal busy error.
            # An implementation counting only pending bytes would acknowledge c.
            pressure.wait()
            assert isinstance(pressure.error, DatabaseBusy), repr(pressure.error)
            flush.wait()
            assert isinstance(flush.error, DatabaseBusy), repr(flush.error)
            assert vault._native().cache_stats() == (2, 0, 62)
            assert vault.get(b"c") is None
            assert vault[b"a"] == b"a" * 40
            assert vault[b"b"] == b"b" * 20
        assert vault.flush_cache() == 2
        vault[b"c"] = b"c" * 10
        vault.flush_cache()
        _committed(path, {b"a": b"a" * 40, b"b": b"b" * 20, b"c": b"c" * 10})


def test_injected_flush_failure_retains_newer_pending_overlay_and_retry(tmp_path):
    _run("failure", tmp_path / "failure.db")


def _failure(path):
    with KVault(path, retries=1) as vault:
        vault["changing"] = b"committed-old"
        _sql(
            path,
            "CREATE TRIGGER reject_flush BEFORE INSERT ON kv BEGIN "
            "SELECT RAISE(ABORT, 'injected flush failure'); END;",
        )
        _enable(vault)
        vault["changing"] = b"inflight-old"
        vault["retained"] = b"keep"
        with _sqlite_writer_lock(path):
            flush = _Worker(vault.flush_cache)
            _inflight(vault, 2)
            _Worker(lambda: vault.put("changing", b"newer-overlay")).result()
            _Worker(lambda: vault.put("late", b"late-value")).result()
            assert vault._native().cache_stats()[:2] == (2, 2)
            assert not flush.done.is_set()
        flush.wait()
        assert flush.error is not None
        assert "injected flush failure" in str(flush.error)
        assert vault._native().cache_stats()[:2] == (3, 0)
        expected = {"changing": b"newer-overlay", "retained": b"keep", "late": b"late-value"}
        for key, value in expected.items():
            assert vault[key] == value
        _committed(path, {"changing": b"committed-old", "retained": None, "late": None})
        _sql(path, "DROP TRIGGER reject_flush;")
        assert vault.flush_cache() == 3
        _committed(path, expected)
        assert vault._native().cache_stats() == (0, 0, 0)


@pytest.mark.parametrize("operation", ["direct", "stream", "delete", "bulk"])
@pytest.mark.parametrize("stage", ["pending", "inflight"])
def test_later_mutation_cannot_resurrect_pending_or_inflight(tmp_path, operation, stage):
    _run("ordering", tmp_path / "ordering.db", operation, stage)


def _ordering(path, operation, stage):
    with KVault(path, retries=1) as vault:
        vault["key"] = b"committed-old"
        _enable(vault, 256)
        vault["key"] = b"buffered-old"
        vault["unrelated"] = b"keep"
        # A value larger than the cache capacity exercises the direct bypass.
        newer = b"new-direct" * 64 if operation == "direct" else b"new-stream"

        def mutate():
            if operation == "direct":
                vault.put("key", newer)
            elif operation == "bulk":
                vault.put_many([("key", newer)])
            elif operation == "stream":
                vault.put_file("key", io.BytesIO(newer), size=len(newer), chunk_size=3)
            else:
                assert vault.delete("key")

        if stage == "inflight":
            with _sqlite_writer_lock(path):
                flush = _Worker(vault.flush_cache)
                _inflight(vault, 2)
                _Worker(lambda: vault.put("key", b"pending-overlay")).result()
                mutation = _Worker(mutate)
                assert not flush.done.is_set()
            flush.result()
            mutation.result()
        else:
            mutate()
        expected = None if operation == "delete" else newer
        assert vault.get("key") == expected
        assert vault["unrelated"] == b"keep"
        vault.flush_cache()
        assert vault.get("key") == expected
        _committed(path, {"key": expected, "unrelated": b"keep"})
        vault.disable_cache()
        _committed(path, {"key": expected, "unrelated": b"keep"})


@pytest.mark.parametrize("operation", ["disable", "reset", "close"])
def test_lifecycle_transition_resolves_concurrent_flush_and_pending_overlay(tmp_path, operation):
    _run("lifecycle", tmp_path / "lifecycle.db", operation)


def _lifecycle(path, operation):
    vault = KVault(path, retries=1)
    try:
        vault["committed"] = b"persisted"
        _enable(vault)
        vault["key"] = b"inflight-old"
        vault["retained"] = b"keep"
        with _sqlite_writer_lock(path):
            flush = _Worker(vault.flush_cache)
            _inflight(vault, 2)
            _Worker(lambda: vault.put("key", b"pending-new")).result()
            _Worker(lambda: vault.put("late", b"accepted")).result()
            transition = _Worker(
                {
                    "disable": vault.disable_cache,
                    "reset": lambda: _enable(vault, 4096),
                    "close": vault.close,
                }[operation]
            )
            assert _Worker(lambda: vault["committed"]).result() == b"persisted"
            assert _Worker(lambda: vault["key"]).result() == b"pending-new"
            assert not flush.done.is_set()
            assert not transition.done.is_set()
        flush.result()
        transition.result()
        expected = {"key": b"pending-new", "retained": b"keep", "late": b"accepted"}
        _committed(path, expected)
        if operation != "close":
            for key, value in expected.items():
                assert vault[key] == value
            assert vault._native().cache_stats() == (0, 0, 0)
            vault["after"] = b"usable"
            vault.flush_cache()
            _committed(path, {"after": b"usable"})
        else:
            with pytest.raises(Exception, match="closed"):
                vault.put("after", b"must-not-be-accepted")
            _committed(path, {"after": None})
    finally:
        vault.close()


def test_slow_stream_sink_keeps_snapshot_without_blocking_same_handle_writer(tmp_path):
    _run("stream_snapshot", tmp_path / "stream-snapshot.db")


def _stream_snapshot(path):
    entered = threading.Event()
    release = threading.Event()
    original = b"original-stream" * 100

    class SlowSink(io.BytesIO):
        def write(self, value):
            entered.set()
            assert release.wait(_DEADLINE), "sink was not released"
            return super().write(value)

    with KVault(path, retries=1) as vault:
        vault["stream"] = original
        sink = SlowSink()
        stream = _Worker(lambda: vault.get_to_file("stream", sink, chunk_size=31))
        try:
            assert entered.wait(_DEADLINE), "stream did not reach sink"
            _Worker(lambda: vault.put("stream", b"replacement")).result()
            assert not stream.done.is_set()
            assert vault["stream"] == b"replacement"
        finally:
            release.set()
        assert stream.result() == len(original)
        assert sink.getvalue() == original
        _committed(path, {"stream": b"replacement"})


if __name__ == "__main__":
    {
        "visibility": _visibility,
        "capacity": _capacity,
        "failure": _failure,
        "ordering": _ordering,
        "lifecycle": _lifecycle,
        "stream_snapshot": _stream_snapshot,
    }[sys.argv[1]](*sys.argv[2:])
