# KohakuVault Documentation

Use this index to navigate the guides that ship with the repository. Each page is versioned with the code in this tree, so the APIs, defaults, and caveats match what you have locally.

## Quick Starts
- [README](../README.md) – overview, installation, and end-to-end demo that stitches `KVault`, `ColumnVault`, and `VectorKVault` together.
- [`examples/basic_usage.py`](../examples/basic_usage.py) – the fastest way to see the Python API without reading every guide.

## Core Guides
- [KVault Guide](kvault.md) – dict-like storage, auto-packing, caching, streaming, and retry semantics.
- [TextVault Guide](textvault.md) – full-text search with FTS5 BM25 ranking, ideal for RAG pipelines.
- [ColumnVault Guide](columnvault.md) – typed columns, dtype grammar, structured data, caching, and chunk tuning.
- [Vector Storage & Search](vectors.md) – vector dtypes inside `ColumnVault` plus similarity search with `VectorKVault`/`sqlite-vec`.
- [Structured Data Cookbook](structured_columns.md) – practical recipes for MessagePack/CBOR columns, schema validation, and hybrid layouts.
- [DataPacker Reference](datapacker.md) – dtype strings, batch APIs, JSON-Schema validation, and how auto-pack builds on it.

## System Architecture & Releases
- [Architecture Deep Dive](architecture.md) – Python↔Rust layering, SQLite layout, cache design, and the auto-pack pipeline.
- [Release Notes](../RELEASE_NOTES_0.7.0.md) – highlights for the 0.7.x series, including vector search and auto-packing.

## Testing & Development
- [tests/](../tests) – pytest suite covering KV, columns, vectors, and structured data.
- [examples/](../examples) – benchmark and tutorial scripts you can run to validate performance claims.

If you add a new component, drop a short guide in this folder and link it here so future readers know where to look.
