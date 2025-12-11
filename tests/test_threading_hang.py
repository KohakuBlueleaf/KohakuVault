"""
Test reproducing the exact threading hang scenario reported by user.

Scenario: Multiple threads using asyncio.to_thread, each opening ColumnVault
on the same file and calling len(cv["key"]) - this hangs reliably.
"""

import asyncio
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from kohakuvault import ColumnVault

# Skip threading tests on Windows CI - temp directory cleanup issues
SKIP_ON_WINDOWS_CI = sys.platform == "win32" and os.environ.get("CI") == "true"


@pytest.mark.skipif(SKIP_ON_WINDOWS_CI, reason="Temp dir issues on Windows CI")
def test_threading_simultaneous_len_check():
    """
    Reproduce the exact user scenario:
    - Multiple threads opening same db file
    - Each calling len(cv["column"])
    - Should hang if GIL not released
    """
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "metrics.db")

        # Create database with data (like user's metrics)
        print(f"Creating database at {db_path}...")
        cv = ColumnVault(db_path)
        col = cv.create_column("step", "i64")
        for i in range(1000):
            col.append(i)
        cv.checkpoint()
        del cv
        print(f"Database created with 1000 entries\n")

        results = {}
        errors = {}
        lock = threading.Lock()
        barrier = threading.Barrier(3)  # All 3 threads wait here

        def thread_worker(thread_id):
            """Worker that opens DB and checks length - matches user's code."""
            try:
                # Wait for all threads to be ready
                print(f"[Thread {thread_id}] Waiting at barrier...")
                barrier.wait()

                # Now all threads execute simultaneously
                start_time = time.time()
                print(f"[Thread {thread_id}] {start_time:.6f} - Opening ColumnVault...")

                cv = ColumnVault(str(db_path))
                open_time = time.time()
                print(
                    f"[Thread {thread_id}] {open_time:.6f} - Opened! Took {open_time - start_time:.6f}s"
                )

                # This is where the hang occurs according to user
                print(f"[Thread {thread_id}] {time.time():.6f} - Calling len(cv['step'])...")
                metrics_count = len(cv["step"])
                len_time = time.time()
                print(
                    f"[Thread {thread_id}] {len_time:.6f} - Got length={metrics_count}, took {len_time - open_time:.6f}s"
                )

                with lock:
                    results[thread_id] = {
                        "count": metrics_count,
                        "open_duration": open_time - start_time,
                        "len_duration": len_time - open_time,
                        "total_duration": len_time - start_time,
                    }

                del cv
                print(f"[Thread {thread_id}] {time.time():.6f} - Completed\n")

            except Exception as e:
                with lock:
                    errors[thread_id] = str(e)
                print(f"[Thread {thread_id}] ERROR: {e}\n")
                import traceback

                traceback.print_exc()

        # Launch threads
        print("=" * 70)
        print("Launching 3 threads to access same database simultaneously...")
        print("=" * 70 + "\n")

        threads = []
        for i in range(3):
            t = threading.Thread(target=thread_worker, args=(i,), name=f"Worker-{i}")
            threads.append(t)
            t.start()

        # Wait for completion with timeout
        timeout = 10  # Should complete quickly, not hang
        print(f"Waiting for threads to complete (timeout={timeout}s)...\n")

        start_wait = time.time()
        for i, t in enumerate(threads):
            remaining_timeout = max(0.1, timeout - (time.time() - start_wait))
            t.join(timeout=remaining_timeout)
            if t.is_alive():
                print(f"\n{'!' * 70}")
                print(f"❌ HANG DETECTED: Thread {i} still alive after {timeout}s!")
                print(f"{'!' * 70}\n")
                print("This confirms the GIL deadlock issue.")
                print("All threads are blocked waiting for SQLite operations to complete.")
                raise AssertionError(f"Thread {i} hung - GIL not released during SQLite operations")

        # Check results
        print("=" * 70)
        print("Results:")
        print("=" * 70)

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        for tid, result in sorted(results.items()):
            print(f"Thread {tid}:")
            print(f"  Count: {result['count']}")
            print(f"  Open time: {result['open_duration']:.6f}s")
            print(f"  len() time: {result['len_duration']:.6f}s")
            print(f"  Total: {result['total_duration']:.6f}s")

        print("\n✅ All threads completed without hanging!")
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


async def async_check_metrics(db_path, task_id):
    """Async task that checks metrics - user's actual scenario."""

    def sync_check():
        """The synchronous part that runs in thread pool."""
        start = time.time()
        print(
            f"[Task {task_id}] {start:.6f} - Starting in thread {threading.current_thread().name}"
        )

        cv = ColumnVault(str(db_path))
        open_time = time.time()
        print(f"[Task {task_id}] {open_time:.6f} - Opened CV")

        metrics_count = len(cv["step"])  # This hangs
        len_time = time.time()
        print(f"[Task {task_id}] {len_time:.6f} - Got len={metrics_count}")

        del cv
        return metrics_count, len_time - start

    # Run in thread pool (asyncio.to_thread)
    result = await asyncio.to_thread(sync_check)
    return result


@pytest.mark.skipif(SKIP_ON_WINDOWS_CI, reason="Temp dir issues on Windows CI")
@pytest.mark.asyncio
async def test_asyncio_gather_scenario():
    """
    Test the exact user scenario: asyncio.gather + asyncio.to_thread
    Multiple async tasks all opening same db file via to_thread.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "metrics.db"

        # Create database
        print(f"Creating database at {db_path}...")
        cv = ColumnVault(str(db_path))
        col = cv.create_column("step", "i64")
        for i in range(500):
            col.append(i)
        cv.checkpoint()
        del cv
        print(f"Database created\n")

        # Simulate user's code: asyncio.gather with multiple tasks
        print("=" * 70)
        print("Running asyncio.gather with 4 concurrent tasks (asyncio.to_thread)...")
        print("=" * 70 + "\n")

        start = time.time()

        # This is what user is doing
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    async_check_metrics(db_path, 0),
                    async_check_metrics(db_path, 1),
                    async_check_metrics(db_path, 2),
                    async_check_metrics(db_path, 3),
                ),
                timeout=10.0,  # Should not hang
            )

            elapsed = time.time() - start
            print(f"\n✅ All tasks completed in {elapsed:.3f}s")

            for i, (count, duration) in enumerate(results):
                print(f"Task {i}: count={count}, duration={duration:.6f}s")

        except asyncio.TimeoutError:
            print(f"\n{'!' * 70}")
            print(f"❌ TIMEOUT: Tasks hung after 10s!")
            print(f"{'!' * 70}")
            raise AssertionError("asyncio.gather hung - GIL deadlock detected")
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


@pytest.mark.skipif(SKIP_ON_WINDOWS_CI, reason="Temp dir issues on Windows CI")
def test_forced_collision():
    """
    Force collision by having threads sleep to align their timing,
    then all call len() at exact same moment.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "test.db")

        # Create database
        cv = ColumnVault(db_path)
        col = cv.create_column("data", "bytes")
        for i in range(100):
            col.append(f"item_{i}".encode())
        del cv

        results = []
        errors = []
        lock = threading.Lock()

        # Use event to synchronize exact timing
        ready_event = threading.Event()
        go_event = threading.Event()

        def delayed_worker(thread_id, delay):
            """Worker with controlled delay to force collision."""
            try:
                # Each thread opens at staggered times
                time.sleep(delay)

                t1 = time.time()
                print(f"[T{thread_id}] {t1:.6f} Opening...")
                cv = ColumnVault(db_path)

                t2 = time.time()
                print(f"[T{thread_id}] {t2:.6f} Opened in {t2-t1:.6f}s, waiting for GO signal...")

                # Signal ready
                with lock:
                    results.append(("ready", thread_id))

                # Wait for GO signal so all threads call len() simultaneously
                go_event.wait(timeout=5)

                t3 = time.time()
                print(f"[T{thread_id}] {t3:.6f} Calling len()...")
                length = len(cv["data"])  # ALL THREADS CALL THIS AT SAME TIME

                t4 = time.time()
                print(f"[T{thread_id}] {t4:.6f} Got length={length} in {t4-t3:.6f}s")

                with lock:
                    results.append(("done", thread_id, length, t4 - t3))

                del cv

            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))
                print(f"[T{thread_id}] ERROR: {e}")

        print("=" * 70)
        print("Test: Force exact collision on len() call")
        print("=" * 70 + "\n")

        # Launch threads with staggered delays
        threads = []
        num_threads = 4

        for i in range(num_threads):
            t = threading.Thread(target=delayed_worker, args=(i, i * 0.05))
            threads.append(t)
            t.start()

        # Wait for all to be ready
        time.sleep(0.5)

        # Count how many are ready
        ready_count = len([r for r in results if r[0] == "ready"])
        print(f"\n{ready_count}/{num_threads} threads ready and holding connections")
        print("Sending GO signal - all will call len() NOW!\n")

        go_event.set()

        # Wait for completion
        timeout = 10
        for i, t in enumerate(threads):
            t.join(timeout=timeout)
            if t.is_alive():
                raise AssertionError(f"Thread {i} HUNG on simultaneous len() call!")

        done_count = len([r for r in results if r[0] == "done"])
        assert done_count == num_threads, f"Only {done_count}/{num_threads} completed"
        assert len(errors) == 0, f"Errors: {errors}"

        print("\n✅ All threads completed simultaneous len() call")
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("THREADING HANG REPRODUCTION TESTS")
    print("=" * 70 + "\n")

    print("[TEST 1] Direct threading with barrier synchronization")
    print("-" * 70)
    test_threading_simultaneous_len_check()

    print("\n[TEST 2] Force exact collision timing")
    print("-" * 70)
    test_forced_collision()

    print("\n[TEST 3] asyncio.gather + asyncio.to_thread (user's exact scenario)")
    print("-" * 70)
    asyncio.run(test_asyncio_gather_scenario())

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - No hangs detected")
    print("=" * 70)
