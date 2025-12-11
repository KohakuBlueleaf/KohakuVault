"""
Test for SWMR (Single Writer Multiple Reader) scenarios.

This test reproduces the hanging issue when multiple threads/processes
try to access the same ColumnVault database file simultaneously.
"""

import multiprocessing
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from kohakuvault import ColumnVault

# Skip multiprocessing tests on Windows CI - temp directory cleanup issues
SKIP_MP_ON_WINDOWS = sys.platform == "win32" and os.environ.get("CI") == "true"


@pytest.mark.skipif(SKIP_MP_ON_WINDOWS, reason="Temp dir issues on Windows CI")
def test_multiple_threads_same_process():
    """Test multiple threads opening and reading from same database."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "test.db")

        # Create database and populate it
        cv = ColumnVault(db_path)
        col = cv.create_column("test_col", "i64")
        for i in range(100):
            col.append(i)
        cv.checkpoint()
        del cv  # Close connection

        results = []
        errors = []

        def reader_thread(thread_id):
            """Thread that opens DB and reads length."""
            try:
                # Each thread opens its own connection
                cv = ColumnVault(db_path)
                col = cv["test_col"]

                # This should NOT hang
                length = len(col)
                results.append((thread_id, length))

                # Read some data
                value = col[0]
                assert value == 0

                del cv
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Launch multiple threads simultaneously
        threads = []
        num_threads = 5

        for i in range(num_threads):
            t = threading.Thread(target=reader_thread, args=(i,))
            threads.append(t)

        # Start all threads at roughly the same time
        for t in threads:
            t.start()

        # Wait for all threads with timeout
        timeout = 10  # Should complete in seconds, not hang forever
        for t in threads:
            t.join(timeout=timeout)
            if t.is_alive():
                pytest.fail(f"Thread hung! Still alive after {timeout}s timeout")

        # Verify all threads completed successfully
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"

        # All should read the same length
        for thread_id, length in results:
            assert length == 100, f"Thread {thread_id} read wrong length: {length}"
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def reader_process(db_path, process_id, result_queue):
    """Process that opens DB and reads length."""
    try:
        cv = ColumnVault(db_path)
        col = cv["test_col"]

        # This should NOT hang
        length = len(col)

        # Read some data
        value = col[0]

        result_queue.put(("success", process_id, length, value))
    except Exception as e:
        result_queue.put(("error", process_id, str(e)))


@pytest.mark.skipif(SKIP_MP_ON_WINDOWS, reason="Multiprocessing temp dir issues on Windows CI")
def test_multiple_processes_reading():
    """Test multiple processes opening and reading from same database."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "test.db")

        # Create database and populate it
        cv = ColumnVault(db_path)
        col = cv.create_column("test_col", "i64")
        for i in range(100):
            col.append(i)
        cv.checkpoint()
        del cv  # Close connection

        # Launch multiple processes
        num_processes = 3
        processes = []
        result_queue = multiprocessing.Queue()

        for i in range(num_processes):
            p = multiprocessing.Process(target=reader_process, args=(db_path, i, result_queue))
            processes.append(p)

        # Start all processes
        for p in processes:
            p.start()

        # Wait for all processes with timeout
        timeout = 10
        for p in processes:
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
                p.join()
                pytest.fail(f"Process hung! Had to terminate after {timeout}s")

        # Collect results
        results = []
        errors = []
        while not result_queue.empty():
            result = result_queue.get()
            if result[0] == "success":
                _, proc_id, length, value = result
                results.append((proc_id, length, value))
            else:
                _, proc_id, error = result
                errors.append((proc_id, error))

        # Verify all processes completed successfully
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert (
            len(results) == num_processes
        ), f"Expected {num_processes} results, got {len(results)}"

        for proc_id, length, value in results:
            assert length == 100, f"Process {proc_id} read wrong length: {length}"
            assert value == 0, f"Process {proc_id} read wrong value: {value}"
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


@pytest.mark.skipif(SKIP_MP_ON_WINDOWS, reason="Temp dir issues on Windows CI")
def test_concurrent_open_close_stress():
    """Stress test: rapidly open/close connections in multiple threads."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "test.db")

        # Create database
        cv = ColumnVault(db_path)
        col = cv.create_column("test_col", "i64")
        col.append(42)
        del cv

        errors = []
        success_count = [0]  # Use list to make it mutable in closure
        lock = threading.Lock()

        def open_read_close(thread_id, iterations):
            """Repeatedly open, read, close."""
            try:
                for i in range(iterations):
                    cv = ColumnVault(db_path)
                    col = cv["test_col"]
                    length = len(col)  # This should not hang
                    assert length == 1
                    del cv

                with lock:
                    success_count[0] += iterations
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Launch multiple threads
        threads = []
        num_threads = 4
        iterations_per_thread = 10

        for i in range(num_threads):
            t = threading.Thread(target=open_read_close, args=(i, iterations_per_thread))
            threads.append(t)
            t.start()

        # Wait with timeout
        timeout = 15
        for t in threads:
            t.join(timeout=timeout)
            if t.is_alive():
                pytest.fail(f"Thread hung during stress test!")

        assert len(errors) == 0, f"Errors: {errors}"
        assert success_count[0] == num_threads * iterations_per_thread
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


@pytest.mark.skipif(SKIP_MP_ON_WINDOWS, reason="Temp dir issues on Windows CI")
def test_varsize_column_swmr():
    """Test SWMR with variable-size columns (the original reported issue)."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "test.db")

        # Create database with variable-size column
        cv = ColumnVault(db_path)
        col = cv.create_column("varsize_col", "bytes")  # Variable-size
        for i in range(50):
            col.append(b"test data " + str(i).encode())
        cv.checkpoint()
        del cv

        results = []
        errors = []

        def reader_thread(thread_id):
            """Thread that reads from variable-size column."""
            try:
                cv = ColumnVault(db_path)
                col = cv["varsize_col"]

                # len() on varsize column - this was reported to hang
                length = len(col)
                results.append((thread_id, length))

                # Read first element
                value = col[0]
                assert value == b"test data 0"

                del cv
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Launch threads
        threads = []
        num_threads = 5

        for i in range(num_threads):
            t = threading.Thread(target=reader_thread, args=(i,))
            threads.append(t)
            t.start()

        # Wait with timeout
        timeout = 10
        for t in threads:
            t.join(timeout=timeout)
            if t.is_alive():
                pytest.fail(f"Thread hung on varsize column access!")

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == num_threads
        for thread_id, length in results:
            assert length == 50
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    print("Running SWMR tests...")
    print("\n1. Testing multiple threads in same process...")
    test_multiple_threads_same_process()
    print("✓ Passed")

    print("\n2. Testing multiple processes...")
    test_multiple_processes_reading()
    print("✓ Passed")

    print("\n3. Testing concurrent open/close stress...")
    test_concurrent_open_close_stress()
    print("✓ Passed")

    print("\n4. Testing variable-size column SWMR...")
    test_varsize_column_swmr()
    print("✓ Passed")

    print("\n✅ All SWMR tests passed!")
