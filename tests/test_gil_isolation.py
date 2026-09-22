"""Native SQLite waits must leave the asyncio loop and same-vault readers runnable."""

import asyncio
import io
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from kohakuvault import KVault, TextVault

# Holds a write lock on the database from another process for a fixed window.
_LOCK_HOLDER = """
import sqlite3
import sys
import time

connection = sqlite3.connect(sys.argv[1])
connection.execute("BEGIN IMMEDIATE")
print("locked", flush=True)
time.sleep(0.8)
connection.rollback()
connection.close()
"""


@pytest.mark.parametrize("operation", ["put", "flush", "disable", "lock", "fts", "open"])
async def test_native_wait_releases_gil(tmp_path, operation):
    path = tmp_path / "native.db"
    if operation == "fts":
        vault = TextVault(str(path))
        row_id = vault.insert("existing document", {"value": "old"})
        write = lambda: vault.insert("new document", {"value": "new"})
        read = lambda: vault.get_by_id(row_id)
    else:
        vault = KVault(path)
        vault["existing"] = {"value": "old"}
        if operation in ("flush", "disable", "lock"):
            vault.enable_cache(cap_bytes=1 << 20, flush_threshold=1 << 19, flush_interval=None)
            vault["next"] = {"value": "new"}
            write = vault.disable_cache if operation == "disable" else vault.flush_cache
        elif operation == "open":
            write = lambda: KVault(path, table="additional")
        else:
            write = lambda: vault.put("next", {"value": "new"})
        if operation == "lock":

            def read():
                # Entering the context waits for an in-flight cache transaction.
                with vault.lock_cache():
                    return vault.get("existing")

        else:
            read = lambda: vault.get("existing")
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _LOCK_HOLDER,
        str(path),
        stdout=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert (await child.stdout.readline()).strip() == b"locked"
    gaps = []
    running = True

    async def pulse():
        before = time.perf_counter()
        while running:
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - before)
            before = now

    ticker = asyncio.create_task(pulse())
    await asyncio.sleep(0.02)
    start = time.perf_counter()
    workers = []
    try:
        writer = asyncio.create_task(asyncio.to_thread(write))
        workers.append(writer)
        await asyncio.sleep(0.05)
        reader = asyncio.create_task(asyncio.to_thread(read))
        workers.append(reader)
        result, old = await asyncio.gather(*(asyncio.shield(task) for task in workers))
        elapsed = time.perf_counter() - start
        await asyncio.sleep(0.02)
        running = False
        await ticker
        assert elapsed >= 0.5, "write never encountered the external lock"
        assert max(gaps) < 0.2, f"event loop stalled for {max(gaps):.3f}s during native wait"
        if operation == "fts":
            assert old[1] == {"value": "old"}
            assert vault.get_by_id(result)[1] == {"value": "new"}
        else:
            assert old == {"value": "old"}
            if operation != "open":
                assert vault["next"] == {"value": "new"}
    finally:
        running = False
        await ticker
        outcomes = await asyncio.gather(*workers, return_exceptions=True)
        await asyncio.wait_for(child.communicate(), timeout=5)
        if operation == "open" and outcomes and isinstance(outcomes[0], KVault):
            outcomes[0].close()
        if operation == "fts":
            del vault._vault
        else:
            vault.close()


def test_concurrent_cache_pressure_preserves_every_acknowledged_write(tmp_path):
    path = tmp_path / "cache-pressure.db"
    vault = KVault(path)
    vault.enable_cache(cap_bytes=512, flush_threshold=512, flush_interval=None)

    def write_batch(worker):
        for number in range(200):
            vault[f"{worker}:{number}"] = b"v" * 128

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write_batch, range(8)))
        vault.flush_cache()
        missing = [
            f"{w}:{n}" for w in range(8) for n in range(200) if vault.get(f"{w}:{n}") != b"v" * 128
        ]
        assert missing == []
    finally:
        vault.close()


def test_full_locked_cache_reports_failure_without_hanging():
    script = """
from kohakuvault import KVault
from kohakuvault.errors import DatabaseBusy
v = KVault(":memory:")
v.enable_cache(cap_bytes=200, flush_threshold=200, flush_interval=None)
with v.lock_cache():
    v["first"] = b"v" * 128
    try:
        v["second"] = b"v" * 128
    except DatabaseBusy as exc:
        assert "locked" in str(exc)
    else:
        raise AssertionError("full locked cache acknowledged an unrecorded write")
assert v["first"] == b"v" * 128
v.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=3,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("cached", [False, True])
def test_stream_callbacks_roundtrip_owned_bytes(tmp_path, cached):
    vault = KVault(tmp_path / "stream.db")
    content = bytes(range(256)) * 32
    try:
        if cached:
            vault.enable_cache(cap_bytes=1 << 20, flush_threshold=1 << 19, flush_interval=None)
            vault["payload"] = content
        else:
            vault.put_file("payload", io.BytesIO(content), size=len(content), chunk_size=37)
        target = io.BytesIO()
        assert vault.get_to_file("payload", target, chunk_size=113) == len(content)
        assert target.getvalue() == content
    finally:
        vault.close()
