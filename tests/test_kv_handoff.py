"""Deterministic regressions for the five P0 failures in local-agent-handoff.md."""

import io
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from kohakuvault import KVault
from kohakuvault.errors import DatabaseBusy
from kohakuvault.proxy import _stream_copy

PAYLOAD = b"abcdefgh"


class ScriptedReader:
    """Bound callbacks so malformed-input regressions cannot loop forever."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def read(self, size):
        self.calls.append(size)
        if not self.responses:
            raise AssertionError("unexpected repeated reader callback")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class PartialSink:
    def __init__(self, accepted):
        self.accepted = accepted
        self.data = bytearray()
        self.calls = 0

    def write(self, value):
        self.calls += 1
        assert self.calls <= 32, "sink made no bounded progress"
        count = min(self.accepted, len(value))
        self.data.extend(value[:count])
        return count


class InvalidSink:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def write(self, value):
        self.calls += 1
        assert self.calls <= 2, "invalid sink result was retried"
        if self.result == "oversized":
            return len(value) + 1
        if self.result == "exception":
            raise OSError("injected sink failure")
        return self.result


def _enable_cache(vault, capacity=1024):
    vault.enable_cache(cap_bytes=capacity, flush_threshold=capacity)


def _copy(vault, source, sink):
    if source == "helper":
        return _stream_copy(io.BytesIO(PAYLOAD), sink, chunk_size=4)
    if source == "cache":
        _enable_cache(vault)
    vault["key"] = PAYLOAD
    return vault.get_to_file("key", sink, chunk_size=4)


@pytest.mark.parametrize("source", ["database", "cache", "helper"])
@pytest.mark.parametrize("accepted", [1, 2, 3])
def test_sink_partial_writes_complete_every_suffix(tmp_path, source, accepted):
    sink = PartialSink(accepted)
    with KVault(tmp_path / "partial.db", retries=1) as vault:
        assert _copy(vault, source, sink) == len(PAYLOAD)
        assert bytes(sink.data) == PAYLOAD


@pytest.mark.parametrize("source", ["database", "cache", "helper"])
@pytest.mark.parametrize(
    "result",
    [None, 0, -1, "oversized", 1.5, "invalid", "exception"],
    ids=["none", "zero", "negative", "oversized", "float", "string", "exception"],
)
def test_sink_invalid_result_is_rejected_without_retry(tmp_path, source, result):
    sink = InvalidSink(result)
    with KVault(tmp_path / "invalid-sink.db", retries=1) as vault:
        with pytest.raises(Exception):
            _copy(vault, source, sink)
        assert sink.calls == 1


@pytest.mark.parametrize("existing", [False, True], ids=["new-key", "overwrite"])
@pytest.mark.parametrize(
    "failure", ["short", "callback", "oversized-read", "oversized-total", "none", "text"]
)
def test_failed_stream_does_not_publish_partial_value(tmp_path, existing, failure):
    responses = {
        "short": [b"abc", b""],
        "callback": [b"abcd", OSError("injected reader failure")],
        "oversized-read": [b"abcde", b"fgh"],
        "oversized-total": [b"012345678"],
        "none": [b"abcd", None],
        "text": [b"abcd", "not bytes"],
    }
    reader = ScriptedReader(*responses[failure])
    path = tmp_path / "failed-stream.db"
    with KVault(path, retries=1) as vault:
        if existing:
            vault["key"] = b"original"
        with pytest.raises(Exception):
            vault.put_file("key", reader, size=8, chunk_size=4)
        assert len(reader.calls) <= 2
        assert vault.get("key") == (b"original" if existing else None)
        with KVault(path, read_only=True, retries=1) as observer:
            assert observer.get("key") == (b"original" if existing else None)
        vault["after-error"] = b"usable"
        assert vault["after-error"] == b"usable"


@pytest.mark.parametrize("chunk_size", [0, -1])
@pytest.mark.parametrize("existing", [False, True])
def test_invalid_stream_chunk_rejected_before_reader(tmp_path, chunk_size, existing):
    reader = ScriptedReader(PAYLOAD)
    with KVault(tmp_path / "invalid-chunk.db", retries=1) as vault:
        if existing:
            vault["key"] = b"original"
        with pytest.raises(Exception):
            vault.put_file("key", reader, size=8, chunk_size=chunk_size)
        assert reader.calls == []
        assert vault.get("key") == (b"original" if existing else None)


@pytest.mark.parametrize("existing", [False, True])
def test_negative_stream_size_rejected_before_reader(tmp_path, existing):
    reader = ScriptedReader(PAYLOAD)
    with KVault(tmp_path / "invalid-size.db", retries=1) as vault:
        if existing:
            vault["key"] = b"original"
        with pytest.raises(Exception):
            vault.put_file("key", reader, size=-1, chunk_size=4)
        assert reader.calls == []
        assert vault.get("key") == (b"original" if existing else None)


@pytest.mark.parametrize("source", ["database", "cache", "helper"])
@pytest.mark.parametrize("chunk_size", [0, -1])
def test_invalid_output_chunk_rejected_before_sink(tmp_path, source, chunk_size):
    sink = PartialSink(2)
    with KVault(tmp_path / "invalid-output-chunk.db", retries=1) as vault:
        if source == "cache":
            _enable_cache(vault)
        vault["key"] = PAYLOAD
        with pytest.raises(Exception):
            if source == "helper":
                _stream_copy(io.BytesIO(PAYLOAD), sink, chunk_size=chunk_size)
            else:
                vault.get_to_file("key", sink, chunk_size=chunk_size)
        assert sink.calls == 0


@pytest.mark.parametrize("existing", [False, True])
def test_empty_stream_publishes_empty_value_without_reading(tmp_path, existing):
    reader = ScriptedReader()
    path = tmp_path / "empty.db"
    with KVault(path, retries=1) as vault:
        if existing:
            vault["key"] = b"original"
        vault.put_file("key", reader, size=0, chunk_size=4)
        assert reader.calls == []
        assert vault["key"] == b""
        sink = PartialSink(2)
        assert vault.get_to_file("key", sink, chunk_size=4) == 0
        assert sink.calls == 0
        with KVault(path, read_only=True, retries=1) as observer:
            assert observer["key"] == b""


@pytest.mark.parametrize("existing", [False, True])
def test_stream_supersedes_pending_value_before_and_after_flush(tmp_path, existing):
    path = tmp_path / "stream-order.db"
    with KVault(path, retries=1) as vault:
        if existing:
            vault["key"] = b"committed-old"
        _enable_cache(vault)
        vault["key"] = b"pending-old"
        vault["unrelated"] = b"keep"
        with KVault(path, read_only=True, retries=1) as observer:
            assert observer.get("key") == (b"committed-old" if existing else None)
            vault.put_file("key", io.BytesIO(b"new-stream"), size=10, chunk_size=4)
            assert vault["key"] == b"new-stream"
            assert observer["key"] == b"new-stream"
            vault.flush_cache()
            assert vault["key"] == b"new-stream"
            assert observer["key"] == b"new-stream"
            assert observer["unrelated"] == b"keep"


@pytest.mark.parametrize("existing", [False, True])
def test_failed_stream_preserves_previous_pending_value(tmp_path, existing):
    path = tmp_path / "failed-cached-stream.db"
    with KVault(path, retries=1) as vault:
        if existing:
            vault["key"] = b"committed-old"
        _enable_cache(vault)
        vault["key"] = b"pending-old"
        vault["unrelated"] = b"keep"
        with pytest.raises(Exception):
            vault.put_file("key", io.BytesIO(b"abc"), size=8, chunk_size=4)
        assert vault["key"] == b"pending-old"
        assert vault["unrelated"] == b"keep"
        with KVault(path, read_only=True, retries=1) as observer:
            # An implementation may commit the preceding cache batch first,
            # but must never expose the failed stream's partial replacement.
            assert observer.get("key") in (
                b"committed-old" if existing else None,
                b"pending-old",
            )
        vault.flush_cache()
        with KVault(path, read_only=True, retries=1) as observer:
            assert observer["key"] == b"pending-old"
            assert observer["unrelated"] == b"keep"


@pytest.mark.parametrize("state", ["same-pending", "same-committed", "unrelated-pending"])
def test_oversized_put_under_cache_lock_rejected_without_mutation(tmp_path, state):
    path = tmp_path / "oversized-locked.db"
    with KVault(path, retries=1) as vault:
        if state == "same-committed":
            vault["key"] = b"committed-old"
        _enable_cache(vault, capacity=64)
        if state == "same-pending":
            vault["key"] = b"pending-old"
        elif state == "unrelated-pending":
            vault["other"] = b"keep"
        previous = vault.get("key")
        with KVault(path, read_only=True, retries=1) as observer:
            committed = observer.get("key")
            with vault.lock_cache():
                with pytest.raises(DatabaseBusy):
                    vault["key"] = b"x" * 128
                assert vault.get("key") == previous
                assert observer.get("key") == committed
                if state == "unrelated-pending":
                    assert vault["other"] == b"keep"
                    assert observer.get("other") is None
            vault.flush_cache()
            assert observer.get("key") == previous
            if state == "unrelated-pending":
                assert observer["other"] == b"keep"


def test_nested_cache_lock_does_not_unlock_outer_context(tmp_path):
    path = tmp_path / "nested-lock.db"
    with KVault(path, retries=1) as vault:
        _enable_cache(vault)
        with KVault(path, read_only=True, retries=1) as observer:
            with vault.lock_cache():
                vault["outer"] = b"one"
                with vault.lock_cache():
                    vault["inner"] = b"two"
                    assert vault.flush_cache() == 0
                assert vault.flush_cache() == 0
                assert observer.get("outer") is None
                assert observer.get("inner") is None
                with pytest.raises(DatabaseBusy):
                    vault.close()
                assert vault["outer"] == b"one"
            assert vault.flush_cache() == 2
            assert observer["outer"] == b"one"
            assert observer["inner"] == b"two"


def test_close_under_cache_lock_rejected_and_retryable(tmp_path):
    path = tmp_path / "close-locked.db"
    with KVault(path, retries=1) as vault:
        _enable_cache(vault)
        with vault.lock_cache():
            vault["pending"] = b"keep"
            with pytest.raises(DatabaseBusy):
                vault.close()
            assert vault["pending"] == b"keep"
            vault["after-rejection"] = b"also-keep"
        vault.close()
        vault.close()
    with KVault(path, read_only=True, retries=1) as observer:
        assert observer["pending"] == b"keep"
        assert observer["after-rejection"] == b"also-keep"


def _run_bounded_child(tmp_path, scenario, *arguments):
    # A native lock regression may retain the GIL, so a thread timeout alone
    # cannot protect the test runner. subprocess.run kills timed-out children.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), scenario, str(tmp_path / "child.db")]
        + list(arguments),
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("source", ["reader", "database-sink", "cache-sink"])
@pytest.mark.parametrize("operation", ["get", "put", "flush", "close"])
def test_stream_callback_reentry_is_rejected_without_deadlock(tmp_path, source, operation):
    _run_bounded_child(tmp_path, "reentry", source, operation)


def _callback_reentry_scenario(path, source, operation):
    with KVault(path, retries=1) as vault:
        vault["stream"] = PAYLOAD
        if source == "cache-sink":
            _enable_cache(vault)
            vault["stream"] = PAYLOAD
        calls = []

        def reenter():
            calls.append(operation)
            assert len(calls) == 1, "failed callback must not be retried"
            if operation == "get":
                vault["stream"]
            elif operation == "put":
                vault["nested"] = b"must-not-publish"
            elif operation == "flush":
                vault.flush_cache()
            else:
                vault.close()

        class ReentrantReader:
            def read(self, size):
                reenter()
                return b"x" * size

        class ReentrantSink:
            def write(self, value):
                reenter()
                return len(value)

        with pytest.raises(Exception, match="reentrant"):
            if source == "reader":
                vault.put_file("stream", ReentrantReader(), size=8, chunk_size=4)
            else:
                vault.get_to_file("stream", ReentrantSink(), chunk_size=4)
        assert calls == [operation]
        assert vault["stream"] == PAYLOAD
        assert vault.get("nested") is None
        vault["after-callback"] = b"usable"
        vault.flush_cache()
        with KVault(path, read_only=True, retries=1) as observer:
            assert observer["stream"] == PAYLOAD
            assert observer.get("nested") is None
            assert observer["after-callback"] == b"usable"


@pytest.mark.parametrize("cache", ["direct", "cached"])
def test_close_fences_put_blocked_in_serialization(tmp_path, cache):
    _run_bounded_child(tmp_path, "serialization-close", cache)


def _serialization_close_scenario(path, cache):
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()
    acknowledged = []
    errors = []
    vault = KVault(path, retries=1)
    if cache == "cached":
        _enable_cache(vault)
    vault["before-close"] = b"keep"

    class BlockingValue:
        def __reduce_ex__(self, protocol):
            entered.set()
            assert release.wait(10), "serialization callback was not released"
            return bytes, (b"late-value",)

    def put():
        try:
            vault["late"] = BlockingValue()
            acknowledged.append(True)
        except Exception as error:
            errors.append(error)
        finally:
            done.set()

    writer = threading.Thread(target=put, daemon=True)
    writer.start()
    try:
        assert entered.wait(5), "put never entered its serialization callback"
        # Close must finish while serialization is suspended, not wait for it.
        vault.close()
        assert not release.is_set()
        assert not done.is_set()
    finally:
        release.set()
        writer.join(5)
    assert done.is_set(), "put remained blocked after callback release"
    assert not acknowledged, "a put serialized before close was acknowledged after close"
    assert len(errors) == 1 and "closed" in str(errors[0]).lower(), errors
    with KVault(path, read_only=True, retries=1) as observer:
        assert observer["before-close"] == b"keep"
        assert observer.get("late") is None
    vault.close()


if __name__ == "__main__":
    if sys.argv[1] == "reentry":
        _callback_reentry_scenario(*sys.argv[2:])
    elif sys.argv[1] == "serialization-close":
        _serialization_close_scenario(*sys.argv[2:])
    else:
        raise ValueError(f"unknown child scenario: {sys.argv[1]}")
