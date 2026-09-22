"""Cache transactions must preserve acknowledged writes under failure and concurrency."""

import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from kohakuvault import KVault
from kohakuvault.errors import DatabaseBusy


def test_failed_flush_preserves_acknowledged_writes(tmp_path):
    path = tmp_path / "failed-flush.db"
    vault = KVault(path)
    vault.enable_cache(cap_bytes=1 << 20, flush_threshold=1 << 20)
    vault["first"] = b"old"
    vault["second"] = b"keep"
    blocker = sqlite3.connect(path)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        with pytest.raises(DatabaseBusy):
            vault.flush_cache()
        blocker.rollback()
        assert vault.get("first") == b"old"
        assert vault.get("second") == b"keep"
        vault["first"] = b"new"
        assert vault.flush_cache() == 2
        assert dict(blocker.execute("SELECT key, value FROM kv")) == {
            b"first": b"new",
            b"second": b"keep",
        }
    finally:
        blocker.rollback()
        blocker.close()
        vault.close()


def test_disable_locked_cache_keeps_pending_writes(tmp_path):
    vault = KVault(tmp_path / "locked-disable.db")
    vault.enable_cache(cap_bytes=1 << 20, flush_threshold=1 << 20)
    try:
        with vault.lock_cache():
            vault["pending"] = b"keep"
            with pytest.raises(DatabaseBusy, match="locked"):
                vault.disable_cache()
            assert vault.get("pending") == b"keep"
        vault.disable_cache()
        assert vault.get("pending") == b"keep"
    finally:
        vault.close()


def _run_in_child(scenario, path):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), scenario, str(path)],
        capture_output=True,
        text=True,
        timeout=25,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_disable_preserves_concurrent_put(tmp_path):
    _run_in_child("disable", tmp_path / "concurrent-disable.db")


def test_waiting_flush_respects_new_cache_lock(tmp_path):
    _run_in_child("lock", tmp_path / "waiting-flush.db")


def _disable_scenario(path):
    first_flushed = threading.Event()
    proceed = threading.Event()
    errors = []
    writer_started = threading.Event()
    writer_done = threading.Event()
    shutdown_done = threading.Event()

    class StagedVault(KVault):
        def flush_cache(self):
            # Schedule another thread between the wrapper's pre-flush and
            # the real native disable_cache operation, without replacing it.
            count = super().flush_cache()
            first_flushed.set()
            assert proceed.wait(5)
            return count

    vault = StagedVault(path)
    vault.enable_cache(cap_bytes=1 << 20, flush_threshold=1 << 20)
    vault["initial"] = b"initial"

    def disable():
        try:
            vault.disable_cache()
        except Exception as exc:
            errors.append(repr(exc))
        finally:
            shutdown_done.set()

    def put():
        writer_started.set()
        try:
            vault["late"] = b"keep"
        except Exception as exc:
            errors.append(repr(exc))
        finally:
            writer_done.set()

    shutdown = threading.Thread(target=disable, daemon=True)
    shutdown.start()
    assert first_flushed.wait(5)
    blocker = sqlite3.connect(path)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        vault["native-batch"] = b"batch"
        proceed.set()
        assert not shutdown_done.wait(0.1)
        writer = threading.Thread(target=put, daemon=True)
        writer.start()
        assert writer_started.wait(5)
        # Both accepting into a cache and waiting for shutdown are legal.
        # In either case a successful put must survive the transition.
        writer_done.wait(0.1)
    finally:
        blocker.rollback()
        blocker.close()
    assert shutdown_done.wait(5)
    assert writer_done.wait(5)
    assert not errors, errors
    assert vault.get("native-batch") == b"batch"
    assert vault.get("late") == b"keep"
    vault.close()


def _lock_scenario(path):
    entered = threading.Event()
    release = threading.Event()
    flush_started = threading.Event()
    flush_done = threading.Event()
    errors = []
    flushed = []

    class HoldingWriter:
        def write(self, value):
            entered.set()
            assert release.wait(5)
            return len(value)

    vault = KVault(path)
    vault["persisted-stream"] = b"stream"
    vault.enable_cache(cap_bytes=1 << 20, flush_threshold=1 << 20)
    vault["before-lock"] = b"before"

    def hold_connection():
        try:
            vault.get_to_file("persisted-stream", HoldingWriter())
        except Exception as exc:
            errors.append(repr(exc))

    def flush():
        flush_started.set()
        try:
            flushed.append(vault.flush_cache())
        except Exception as exc:
            errors.append(repr(exc))
        finally:
            flush_done.set()

    streamer = threading.Thread(target=hold_connection, daemon=True)
    streamer.start()
    assert entered.wait(5)
    flusher = threading.Thread(target=flush, daemon=True)
    flusher.start()
    assert flush_started.wait(5)
    assert not flush_done.wait(0.1)
    with vault.lock_cache():
        vault["inside-lock"] = b"inside"
        release.set()
        assert flush_done.wait(5)
        with sqlite3.connect(path) as observer:
            visible = observer.execute(
                "SELECT value FROM kv WHERE key=?", (b"inside-lock",)
            ).fetchone()
        observer.close()
        assert not errors, errors
        assert flushed == [0]
        assert visible is None, "data from the locked batch was committed before context exit"
    assert vault.flush_cache() == 2
    assert vault["inside-lock"] == b"inside"
    streamer.join(5)
    flusher.join(5)
    vault.close()


if __name__ == "__main__":
    {"disable": _disable_scenario, "lock": _lock_scenario}[sys.argv[1]](sys.argv[2])
