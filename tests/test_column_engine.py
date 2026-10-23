"""Regression tests for the replacement column format and buffer APIs."""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import sqlite3

import numpy as np
import pytest

from kohakuvault import ColumnVault


@pytest.mark.parametrize(
    "dtype", ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "f32", "f64"]
)
def test_numeric_buffers_cross_blocks(tmp_path, dtype):
    with ColumnVault(tmp_path / "data.db", block_bytes=64) as vault:
        col = vault.create_column("numbers", dtype)
        expected = np.arange(
            90, dtype=dtype.replace("i", "int").replace("u", "uint").replace("f", "float")
        )
        col.append_array(expected[:17])
        col.append_array(expected[17:])
        np.testing.assert_array_equal(col.read_array(), expected)
        np.testing.assert_array_equal(col.read_array(7, 40), expected[7:47])
        np.testing.assert_array_equal(col.take([89, 0, 30, 30, -2]), expected[[89, 0, 30, 30, 88]])
        assert list(col) == expected.tolist()
        assert sum(len(batch) for batch in col.iter_batches(batch_bytes=48)) == 90
        with pytest.raises(IndexError):
            col.take([90])
        with pytest.raises(IndexError):
            col.read_array(85, 10)
        assert col.read_array(90, 0).size == 0


def test_tensor_layout_and_owned_read_lifetime(tmp_path):
    path = tmp_path / "tensors.db"
    with ColumnVault(path, block_bytes=64) as vault:
        col = vault.create_column("tensor", "vec:f32:2:3")
        expected = np.arange(120, dtype=">f4").reshape(20, 2, 3)[::2]
        col.append_array(expected)
        result = col.read_array()
        assert result.shape == (10, 2, 3)
        assert not result.flags.writeable
        writable = col.read_array(copy=True)
        writable[0] = 99
        np.testing.assert_array_equal(col[0], expected[0])
        assert len(col.read_buffer()) == expected.size * 4
        with pytest.raises(ValueError):
            col.append_array(np.zeros((3, 6), dtype=np.float32))
    np.testing.assert_array_equal(result, expected)
    path.unlink()


def test_variable_tail_reuse_empty_and_oversized(tmp_path):
    path = tmp_path / "variable.db"
    expected = [b"", b"abc", b"de", b"", b"z" * 100, b"end", b""]
    with ColumnVault(path, block_bytes=32) as vault:
        col = vault.create_column("payload", "bytes")
        for value in expected:
            col.append(value)
        assert col[:] == expected
        payload, offsets = col.read_buffers(1, 5)
        assert [payload[int(a) : int(b)] for a, b in zip(offsets[:-1], offsets[1:])] == expected[
            1:6
        ]
        with sqlite3.connect(path) as conn:
            capacities = [
                row[0] for row in conn.execute("SELECT capacity_bytes FROM kvault_col_blocks")
            ]
        assert all(size == 32 or size == 100 for size in capacities)
        col.clear()
        for _ in range(20):
            col.extend([b"a"])
        # At most two rows per tiny block, not a fresh block per extend().
        assert col.storage_info()["blocks"] == 10
        assert col[:] == [b"a"] * 20


def test_existing_reader_observes_new_commits(tmp_path):
    path = tmp_path / "live.db"
    with ColumnVault(path) as writer:
        col = writer.create_column("numbers", "i64")
        col.extend([1, 2])
        with ColumnVault(path, read_only=True) as reader:
            observed = reader["numbers"]
            assert len(observed) == 2
            with col.cache():
                col.append(3)
                assert len(col) == 3
                assert len(observed) == 2
            assert len(observed) == 3
            assert observed[:] == [1, 2, 3]
            with pytest.raises(PermissionError):
                observed.append(4)
            with pytest.raises(PermissionError):
                reader.create_column("forbidden", "i64")


def test_failed_flush_rolls_back_and_preserves_buffer(tmp_path):
    path = tmp_path / "rollback.db"
    with ColumnVault(path, block_bytes=32) as writer:
        col = writer.create_column("numbers", "i64")
        col.extend([1, 2])
        col.enable_cache(cap_bytes=1024)
        col.extend([3, 4, 5])
        from tests.test_kv_flush_isolation import _sql

        _sql(
            path,
            "CREATE TRIGGER reject_length BEFORE UPDATE OF length ON kvault_col_columns BEGIN SELECT RAISE(ABORT, 'injected'); END;",
        )
        with pytest.raises(RuntimeError, match="injected"):
            col.flush_cache()
        with ColumnVault(path, read_only=True) as reader:
            assert reader["numbers"][:] == [1, 2]
        assert col._pending_rows == 3
        _sql(path, "DROP TRIGGER reject_length;")
        col.flush_cache()
        assert col[:] == [1, 2, 3, 4, 5]


def test_empty_values_are_bounded_in_cache(tmp_path):
    with ColumnVault(tmp_path / "empty.db", block_bytes=64) as vault:
        col = vault.create_column("payload", "bytes")
        with col.cache(cap_bytes=64, flush_threshold=64):
            for _ in range(100):
                col.append(b"")
                assert col._buffer_bytes() <= 64
        assert col[:] == [b""] * 100


def test_new_columns_in_cache_context_and_nested_caches(tmp_path):
    path = tmp_path / "cache.db"
    with ColumnVault(path) as vault:
        with vault.cache(cap_bytes=1024):
            col = vault.create_column("numbers", "i64")
            col.extend([1, 2])
            with col.cache(cap_bytes=64):
                col.append(3)
            col.append(4)
        assert col[:] == [1, 2, 3, 4]
        assert not col._cache_enabled


def test_close_and_deleted_handle(tmp_path):
    path = tmp_path / "close.db"
    vault = ColumnVault(path)
    old = vault.create_column("n", "i64")
    old.extend([1])
    assert vault.delete_column("n")
    fresh = vault.create_column("n", "i64")
    fresh.extend([2])
    with pytest.raises(KeyError):
        old.read_array()
    vault.close()
    with pytest.raises(ValueError, match="closed"):
        fresh.append(3)
    vault.close()
    path.unlink()


def test_reader_opens_while_write_lock_is_held(tmp_path):
    path = tmp_path / "locked.db"
    with ColumnVault(path) as vault:
        vault.create_column("n", "i64").extend([1, 2])
        conn = sqlite3.connect(path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            for readonly in (True, False):
                with ColumnVault(path, read_only=readonly) as reader:
                    assert reader["n"][:] == [1, 2]
        finally:
            conn.rollback()
            conn.close()


def test_same_handle_thread_readers_and_writer(tmp_path):
    with ColumnVault(tmp_path / "threads.db", block_bytes=1024) as vault:
        col = vault.create_column("n", "i64")
        col.extend([0])

        def write():
            for start in range(1, 2001, 100):
                col.append_array(np.arange(start, start + 100, dtype=np.int64))

        def read():
            for _ in range(40):
                array = col.read_array()
                np.testing.assert_array_equal(array, np.arange(len(array)))

        with ThreadPoolExecutor(max_workers=5) as executor:
            tasks = [executor.submit(read) for _ in range(4)]
            tasks.append(executor.submit(write))
            for task in tasks:
                task.result(timeout=15)
        assert len(col) == 2001


def _process_reader(path, ready, proceed, output):
    try:
        with ColumnVault(path, read_only=True) as vault:
            col = vault["n"]
            before = col[:]
            ready.set()
            if not proceed.wait(10):
                raise TimeoutError("writer did not signal")
            output.put((before, col[:]))
    except Exception as error:
        ready.set()
        output.put(repr(error))


def test_independent_process_reader(tmp_path):
    context = multiprocessing.get_context("spawn")
    path = str(tmp_path / "process.db")
    with ColumnVault(path) as writer:
        col = writer.create_column("n", "i64")
        col.extend([1])
        ready, proceed = context.Event(), context.Event()
        output = context.Queue()
        child = context.Process(target=_process_reader, args=(path, ready, proceed, output))
        child.start()
        try:
            assert ready.wait(10)
            col.extend([2, 3])
            proceed.set()
            assert output.get(timeout=10) == ([1], [1, 2, 3])
            child.join(10)
            assert child.exitcode == 0
        finally:
            if child.is_alive():
                child.terminate()
                child.join()
            output.close()


def test_unknown_column_version_refused(tmp_path):
    path = tmp_path / "version.db"
    with ColumnVault(path):
        pass
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE kvault_col_format SET version=999")
    with pytest.raises(ValueError, match="version"):
        ColumnVault(path)


def test_close_rejects_append_still_serializing(tmp_path, monkeypatch):
    from threading import Event

    vault = ColumnVault(tmp_path / "closing.db")
    col = vault.create_column("n", "i64")
    col.enable_cache()
    entered, proceed = Event(), Event()
    pack = col._codec.pack_one

    def delayed_pack(value):
        entered.set()
        assert proceed.wait(10)
        return pack(value)

    monkeypatch.setattr(col._codec, "pack_one", delayed_pack)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(col.append, 7)
        try:
            assert entered.wait(10)
            vault.close()
        finally:
            proceed.set()
        with pytest.raises(ValueError, match="closed"):
            future.result(timeout=10)
    with ColumnVault(tmp_path / "closing.db", read_only=True) as reader:
        assert reader["n"][:] == []
