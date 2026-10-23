"""Independent processes observe whole committed batches without reopening."""

import multiprocessing
import time

from kohakuvault import KVault


def _reader(path, ready, start, results):
    try:
        with KVault(path, read_only=True, reader_cache_kb=256) as vault:
            ready.put(True)
            if not start.wait(15):
                raise AssertionError("writer never started")
            previous = 0
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                first, second = vault.get_many(["version", "mirror"])
                assert first == second, "bulk read crossed commit snapshots"
                assert first >= previous, "committed versions moved backward"
                previous = first
                if first == 100:
                    results.put((True, first))
                    return
            raise AssertionError(f"reader stalled at {previous}")
    except BaseException as error:
        results.put((False, repr(error)))


def test_four_process_readers_observe_atomic_writer_batches(tmp_path):
    path = tmp_path / "processes.db"
    ctx = multiprocessing.get_context("spawn")
    ready, results = ctx.Queue(), ctx.Queue()
    start = ctx.Event()
    with KVault(path) as writer:
        writer.put_many([("version", 0), ("mirror", 0)])
        processes = [
            ctx.Process(target=_reader, args=(str(path), ready, start, results)) for _ in range(4)
        ]
        try:
            for process in processes:
                process.start()
            for _ in processes:
                assert ready.get(timeout=20)
            start.set()
            for version in range(1, 101):
                writer.put_many([("version", version), ("mirror", version)])
            observed = [results.get(timeout=25) for _ in processes]
            assert observed == [(True, 100)] * 4, observed
            for process in processes:
                process.join(10)
                assert process.exitcode == 0
        finally:
            start.set()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
            ready.close()
            results.close()
