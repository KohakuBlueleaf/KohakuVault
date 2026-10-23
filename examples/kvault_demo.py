"""Atomic batches, typed mapping helpers, streaming and independent readers."""

import io
from pathlib import Path
import tempfile

from kohakuvault import KVault, KohakuVaultError


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "demo.db"
        with KVault(path, reader_cache_kb=1024, reader_idle_limit=4) as writer:
            writer.put_many([("config", {"enabled": True}), ("blob", b"original")])
            assert writer.get_many(["config", "missing"]) == [{"enabled": True}, None]
            assert writer.setdefault("tags", ["example"]) == ["example"]
            assert writer.pop("missing", "fallback") == "fallback"
            # Iterables are consumed in committed batches of at most 1024 pairs.
            writer.update(((f"row:{i}", {"id": i}) for i in range(2000)))
            with KVault(path, read_only=True) as reader:
                with writer.cache(cap_bytes=4096, flush_threshold=4096):
                    writer["buffered"] = b"pending"
                    assert reader.get("buffered") is None
                assert reader["buffered"] == b"pending"
                try:
                    writer.put_file("blob", io.BytesIO(b"short"), size=20)
                except KohakuVaultError:
                    pass
                assert reader["blob"] == b"original"
                writer.put_file("blob", io.BytesIO(b"complete"), size=8)
                sink = io.BytesIO()
                assert reader.get_to_file("blob", sink, chunk_size=3) == 8
                assert sink.getvalue() == b"complete"
                print("reader cache:", reader.reader_pool_stats())
            # Inspect this property when using enable_cache(flush_interval=...).
            assert writer.background_flush_error is None
            print("rows:", len(list(writer.items(prefix="row:", batch_size=100))))
            writer.clear()
            assert len(writer) == 0
    print("KV demo passed; all handles closed.")


if __name__ == "__main__":
    main()
