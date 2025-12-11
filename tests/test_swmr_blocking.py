"""
Test for the specific SWMR blocking scenario:
Process A holds connection open, Process B tries to open while A is active.
"""

import multiprocessing
import os
import shutil
import sys
import tempfile
import time

import pytest

from kohakuvault import ColumnVault

# Skip multiprocessing tests on Windows CI - temp directory cleanup issues
SKIP_MP_ON_WINDOWS = sys.platform == "win32" and os.environ.get("CI") == "true"


def process_a_hold_connection(db_path, ready_event, release_event):
    """Process that opens DB and holds it while signaling it's ready."""
    try:
        print(f"Process A: Opening database...")
        cv = ColumnVault(db_path)
        col = cv["test_col"]

        print(f"Process A: Got column, length = {len(col)}")

        # Signal that we're ready (holding the connection)
        ready_event.set()

        # Hold the connection while doing work
        print(f"Process A: Holding connection, waiting for release signal...")
        release_event.wait(timeout=30)

        # Do some work while holding
        for i in range(10):
            time.sleep(0.1)
            _ = len(col)  # Keep accessing

        print(f"Process A: Releasing connection")
        del cv
        print(f"Process A: Done")
    except Exception as e:
        print(f"Process A ERROR: {e}")
        import traceback

        traceback.print_exc()


def process_b_try_open(db_path, start_event, result_queue):
    """Process that tries to open DB after process A has it open."""
    try:
        # Wait for signal to start
        print(f"Process B: Waiting for start signal...")
        start_event.wait(timeout=30)

        print(f"Process B: Attempting to open database...")
        start_time = time.time()

        cv = ColumnVault(db_path)
        col = cv["test_col"]

        elapsed = time.time() - start_time
        print(f"Process B: Successfully opened in {elapsed:.2f}s")

        # Try to read length (this was reported to hang)
        print(f"Process B: Calling len(col)...")
        length = len(col)

        print(f"Process B: Got length = {length}")
        result_queue.put(("success", length, elapsed))

        del cv
        print(f"Process B: Done")
    except Exception as e:
        print(f"Process B ERROR: {e}")
        import traceback

        traceback.print_exc()
        result_queue.put(("error", str(e)))


@pytest.mark.skipif(SKIP_MP_ON_WINDOWS, reason="Multiprocessing temp dir issues on Windows CI")
def test_concurrent_open_with_held_connection():
    """
    Test the specific scenario:
    1. Process A opens DB and holds connection
    2. Process B tries to open while A is holding
    3. B should NOT hang indefinitely (WAL mode should allow this)
    """
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "test.db")

        # Create and populate database
        print("Creating initial database...")
        cv = ColumnVault(db_path)
        col = cv.create_column("test_col", "i64")
        for i in range(100):
            col.append(i)
        cv.checkpoint()
        del cv
        print("Initial database created")

        # Events for coordination
        a_ready = multiprocessing.Event()
        a_release = multiprocessing.Event()
        b_start = multiprocessing.Event()
        result_queue = multiprocessing.Queue()

        # Start Process A (holds connection)
        print("\n--- Starting Process A ---")
        proc_a = multiprocessing.Process(
            target=process_a_hold_connection, args=(db_path, a_ready, a_release)
        )
        proc_a.start()

        # Wait for A to open and signal ready
        assert a_ready.wait(timeout=10), "Process A failed to open database"
        print("Process A is ready and holding connection")

        # Now start Process B while A is holding
        print("\n--- Starting Process B (while A holds connection) ---")
        proc_b = multiprocessing.Process(
            target=process_b_try_open, args=(db_path, b_start, result_queue)
        )
        proc_b.start()

        # Let B start trying to open
        time.sleep(0.5)
        b_start.set()

        # Wait for B to complete (should NOT hang)
        timeout = 15  # Should complete quickly, not hang
        proc_b.join(timeout=timeout)

        if proc_b.is_alive():
            # B is still alive - it hung!
            print(f"\n❌ Process B HUNG after {timeout}s!")
            print("Terminating processes...")
            proc_b.terminate()
            a_release.set()
            proc_a.join(timeout=5)
            if proc_a.is_alive():
                proc_a.terminate()
            proc_a.join()
            proc_b.join()
            pytest.fail(
                f"Process B hung when trying to open database while Process A held connection"
            )

        print("\nProcess B completed, signaling A to release...")
        a_release.set()
        proc_a.join(timeout=10)

        if proc_a.is_alive():
            proc_a.terminate()
            proc_a.join()

        # Check results
        if not result_queue.empty():
            result = result_queue.get()
            if result[0] == "error":
                pytest.fail(f"Process B failed: {result[1]}")
            else:
                _, length, elapsed = result
                print(f"\n✅ Success! Process B read length={length} in {elapsed:.2f}s")
                assert length == 100
        else:
            pytest.fail("No result from Process B")
    finally:
        # Clean up temp directory manually
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def _simultaneous_opener(db_path, proc_id, start_barrier, result_queue):
    """Wait at barrier, then all open simultaneously. Module-level for pickling."""
    try:
        # Wait for all processes to be ready
        start_barrier.wait(timeout=10)

        # Now all processes try to open at the same time
        start_time = time.time()
        cv = ColumnVault(db_path)
        col = cv["test_col"]

        # This is where hang was reported
        length = len(col)

        elapsed = time.time() - start_time

        # Read some data
        value = col[0]

        result_queue.put(("success", proc_id, length, elapsed))
        del cv
    except Exception as e:
        result_queue.put(("error", proc_id, str(e)))


@pytest.mark.skipif(SKIP_MP_ON_WINDOWS, reason="Multiprocessing temp dir issues on Windows CI")
def test_rapid_simultaneous_opens():
    """Multiple processes all try to open at exactly the same time."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "test.db")

        # Create database
        cv = ColumnVault(db_path)
        col = cv.create_column("test_col", "bytes")  # Varsize to match user's scenario
        for i in range(50):
            col.append(f"data_{i}".encode())
        del cv

        start_barrier = multiprocessing.Barrier(5)  # All processes wait here
        result_queue = multiprocessing.Queue()

        # Launch all processes
        processes = []
        num_procs = 5

        for i in range(num_procs):
            p = multiprocessing.Process(
                target=_simultaneous_opener, args=(db_path, i, start_barrier, result_queue)
            )
            processes.append(p)
            p.start()

        # Wait for all with timeout
        timeout = 15
        all_completed = True
        for p in processes:
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
                p.join()
                all_completed = False

        assert all_completed, "Some processes hung during simultaneous open!"

        # Check results
        results = []
        errors = []
        while not result_queue.empty():
            result = result_queue.get()
            if result[0] == "success":
                results.append(result[1:])
            else:
                errors.append(result[1:])

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == num_procs, f"Expected {num_procs} results, got {len(results)}"

        print(f"\n✅ All {num_procs} processes opened simultaneously without hanging")
        for proc_id, length, elapsed in results:
            print(f"  Process {proc_id}: length={length}, time={elapsed:.3f}s")
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    print("=" * 70)
    print("Testing SWMR Blocking Scenarios")
    print("=" * 70)

    print("\n[TEST 1] Concurrent open with held connection...")
    test_concurrent_open_with_held_connection()

    print("\n[TEST 2] Rapid simultaneous opens...")
    test_rapid_simultaneous_opens()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - No hanging detected!")
    print("=" * 70)
