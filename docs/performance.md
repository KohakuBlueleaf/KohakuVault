# Basic performance checks

The column rewrite is measured with `examples/benchmark_columnar.py`, a small smoke benchmark rather than a full database benchmark suite. Build the native extension in release mode before comparing results:

```bash
maturin develop --release
python examples/benchmark_columnar.py --rows 1000000 --json benchmark-results.json
```

The script validates returned values and reports elapsed time, useful payload bytes, and MiB/s. It covers numeric-array append/read, list conversion, 1,000 small slices, buffered scalar appends, and variable-width records. Input construction is outside timing. Cached-write timing includes the final flush. Direct writes commit before returning. Checkpoint time is reported separately.

## Local before/after check

Measured in the KohakuVault Windows workspace with Python 3.13.14 and release native builds. Both engines use SQLite WAL and `synchronous=NORMAL`. This is one warm-process run, not a cold-disk or power-failure durability benchmark. The old engine used its default growing chunks; the new engine uses its default bounded 1 MiB blocks. These numbers compare the complete public paths, including their layout/default changes.

| Workload | Original 0.9.0 | Replacement | Approximate improvement |
|---|---:|---:|---:|
| Extend 1,000,000 i64 values from a Python list | 109.94 ms | 53.26 ms | 2.1x |
| Read 1,000,000 i64 values into a Python list | 42.75 ms | 18.74 ms | 2.3x |
| 1,000 slices of 64 i64 values | 7,611.75 ms | 29.94 ms | 254x |
| Append 10,000 byte records of 64 bytes | 5.02 ms | 4.31 ms | 1.2x |
| Read those byte records into a Python list | 6.93 ms | 2.91 ms | 2.4x |

The slice improvement removes a specific pathological path: the original engine repeatedly materialized a whole multi-megabyte BLOB for a small range. It is not a claim that every operation is 254x faster, or that SQLite disk I/O improved by that factor.

## New buffer paths

| Operation | Payload / count | Measured time | Useful throughput |
|---|---|---:|---:|
| Numeric array append | 1,000,000 i64 values | 53.39 ms | 142.90 MiB/s |
| Numeric array read | 1,000,000 i64 values | 19.67 ms | 387.89 MiB/s |
| Cached scalar append, including flush | 20,000 i64 values | 15.16 ms | 10.06 MiB/s |
| Variable payload/offset read | 10,000 records, 640,000 payload bytes | 0.489 ms | 1,247.14 MiB/s |

The array read happened before the list-read test, so cache state differs. Do not interpret that single run as evidence that list materialization is cheaper than array output. Run repeated, reordered measurements for that question. Variable buffer throughput excludes offset bytes from its useful-byte numerator and avoids constructing individual record objects.

The checkpoint in this run took 9.28 ms. It is not charged to each append. WAL NORMAL can acknowledge commits before a power-loss-durable synchronization; no benchmark result should be described as per-operation fsync throughput.

## What the benchmark does not establish

This script does not measure databases larger than RAM, cold filesystem caches, long-lived snapshots, write endurance, randomized multi-column workloads, compressed storage, or maximum concurrent-reader throughput. Its purpose is to detect obvious regressions in the principal data paths without a large benchmark dependency stack.

Current remaining costs include native-to-Python buffer copying, Python object creation for list/structured-record results, preallocated unused tail space, and a slow whole-column replacement path for structural/variable-width mutations. See [ColumnVault guide](columnvault.md) for API choices and [storage design](columnar_storage.md) for implementation details.

## KV correctness/performance follow-up

Run the same small workload against each package, in separate processes:

```bash
python examples/benchmark_kv.py --structured --json kv-results.json
# Local preserved builds, when available:
python examples/benchmark_kv.py --package-root target/performance-rework/legacy --structured
python examples/benchmark_kv.py --package-root target/performance-rework/current --structured
```

The follow-up comparison used Windows 11, Python 3.13.14, release extensions, 10,000 keys, 256-byte distinct raw payloads, and median of five fresh-database trials. Reads are warm; inputs, validation, and thread startup are outside timers. Cached totals include final commit. WAL and the writer's existing `synchronous=NORMAL` policy are unchanged; explicit TRUNCATE checkpoint time is separate. The checkpoint connection's synchronization setting is not evidence about the writer's setting. Optional `--direct` measures scalar autocommit writes; it was not enabled in the full comparison.

“Pre-follow-up” means the preserved column-rewrite build before streaming/cache correctness and native bulk work. All times below are milliseconds. The JSON artifacts include samples, useful-byte counts, Python/package paths, and extension hashes under ignored `target/performance-rework/kv-review-{original,pre,final}.json`. They are local evidence, not test dependencies.

| Raw workload | Original 0.9.0 | Pre-follow-up | Follow-up |
|---|---:|---:|---:|
| Cached scalar acceptance | 12.350 | 12.025 | 8.971 |
| Final cache flush | 12.960 | 12.816 | 10.717 |
| Cached scalar writes including commit | 25.493 | 25.098 | 19.643 |
| Native `put_many`, committed | unavailable | unavailable | 12.596 |
| Point gets | 47.298 | 43.863 | 30.019 |
| Native `get_many` | unavailable | unavailable | 7.781 |
| Items iteration | 49.668 | 7.463 | 5.297 |
| Four shared-handle readers, 40,000 gets | 529.256 | 264.257 | 111.759 |
| Checkpoint after cached writes | 10.007 | 9.077 | 7.109 |

The complete scalar paths now improve against both comparison builds on this workload, while preserving the ordering/lifecycle protections. Profiling identified redundant Python lock/native policy calls, retry-closure allocation, double-copying raw values, exception-heavy builtin-container detection, GIL round trips for memory-only admission, and explicit BEGIN/ROLLBACK around single SELECTs. These were removed without changing WAL/synchronization policy. Raw batches remain faster than scalar calls; bulk results are not substituted for scalar results.

| Structured workload (dict/list records) | Original 0.9.0 | Pre-follow-up | Follow-up |
|---|---:|---:|---:|
| Cached scalar writes including commit | 40.761 | 39.623 | 24.372 |
| Native `put_many`, committed | unavailable | unavailable | 15.933 |
| Point gets | 59.076 | 54.789 | 40.007 |
| Native `get_many` | unavailable | unavailable | 15.334 |
| Items iteration | 61.019 | 16.057 | 12.666 |
| Four shared-handle readers, 40,000 gets | 688.476 | 651.658 | 376.896 |

Structured useful bytes use compact JSON logical size, not native encoded or disk bytes. The concurrency numbers compare equivalent workloads across builds, not linear scaling; four readers perform four times the point-read work. These sequential local measurements are not a controlled statistical study and do not establish universal speedups. Medians of phases need not sum to the median total.

An intermediate correctness-only build had scalar regressions (raw writes 31.533 ms, point reads 49.681 ms). Those results motivated profiling; the tables above are fresh measurements of all three builds after the optimizations, not a comparison against selectively chosen old timings.

The principal contention improvement is separately established by deterministic tests: while an external process holds SQLite's writer reservation and a flush batch is in-flight, same-handle reads of committed, in-flight, and newer pending values complete before releasing that reservation. Capacity includes both pending and in-flight data; failure and retry retain newest values. These tests measure ordering/isolation, not production latency.

The unchanged column benchmark also ran successfully after the KV follow-up: array append/read 42.562/18.617 ms, list append/read 51.000/17.654 ms, 1,000 small slices 26.124 ms, variable append/list read 4.205/3.093 ms. This smoke run does not replace the original column before/after methodology above.

## Vector comparison with 0.9.0

```bash
python examples/benchmark_vector.py --repeats 5 --json vector-current.json
python examples/benchmark_vector.py --package-root /path/to/old/package-root --repeats 5 --json vector-original.json
```

Same Windows/Python 3.13 release environment, separate processes, median of five fresh-database trials: 2,000 seeded, normalized float32 vectors of dimension 128 with small dict metadata; 200 warm exact cosine queries, top-10. Inserts include commits. Input construction, result validation, worker startup, and cleanup are outside timing. The four-reader row performs 800 queries, not 200. Neither index algorithm nor default WAL/NORMAL policy changes; explicit checkpoint cost is not included. JSON includes every sample and loaded extension hash.

| Workload | Original 0.9.0 | Current | Speedup |
|---|---:|---:|---:|
| 2,000 committed inserts | 245.020 ms | 214.565 ms | 1.14× |
| 2,000 ID lookups | 21.704 ms | 18.399 ms | 1.18× |
| 200 top-10 searches | 116.300 ms | 97.538 ms | 1.19× |
| Four shared-handle readers, 800 searches | 481.983 ms | 276.748 ms | 1.74× |

This measures the complete buffer-conversion, transaction, reader-pool and decoding paths, not an isolated improvement in the sqlite-vec search kernel. Original inserts wrote vector/value separately; current inserts commit both atomically. Local artifacts: `target/performance-rework/vector-{original,final}.json`. Results are workload-specific, not a promise for all dimensions, row counts or hardware.

## Reader policy and memory

Run `python examples/benchmark_readers.py --rows 10000 --repeats 3`. It warms the data, compares native point reads with one-key explicit-snapshot bulk reads, and then runs four readers. The one-key comparison includes bulk dispatch overhead as well as snapshot setup; it does not isolate SQL transaction cost alone. Input construction and result validation are outside timing.

| Cache KiB / mmap bytes / idle limit | 10k point ms | 10k one-key batch ms | 40k four-reader ms | Idle connections | Idle SQLite cache bytes |
|---|---:|---:|---:|---:|---:|
| 128 / 0 / 8 | 36.55 | 47.86 | 165.43 | 4 | 1,389,568 |
| 8192 / 0 / 8 | 41.12 | 48.32 | 153.55 | 4 | 12,023,296 |
| 8192 / 268435456 / 8 | 39.50 | 47.87 | 98.94 | 4 | 266,752 |
| 8192 / 268435456 / 1 | 37.71 | 44.92 | 295.04 | 1 | 66,688 |
| 8192 / 268435456 / 0 | 4118.48 | 4426.25 | 17122.53 | 0 | 0 |

Connection reuse matters: zero idle retention reopens a database for every lease. A single retained connection can cause churn under overlapping reads. Retaining eight idle readers with an 8 MiB per-reader budget and inherited mmap is a reasonable measured default for this workload; applications can reduce the budget for small datasets, or disable mmap as needed. The budgets are SQLite targets, not hard memory caps (page/statement overhead and temporary allocations can exceed them). `SQLITE_DBSTATUS_CACHE_USED` measures actual idle page-cache allocation, **not process RSS, mmap residency, Python objects, active readers or a peak**. Four active readers still remain permitted with idle limit one. No hard active-reader cap is added. Results include median-of-three samples in the local `reader-policy.json` artifact.

## Regression coverage

`tests/test_column_engine.py` checks the new layout, typed buffers, range boundaries, cache failure recovery, read-only access, threaded reads/writes, and an independent-process reader. `tests/test_storage_compatibility.py` uses an actual original-engine database to verify KV/vector compatibility, byte-identical KV rewrites, unlimited pagination, binary prefixes, and atomic vector mutations.

The KV follow-up added streaming/cache/lifecycle failure regressions, short-lock flush contention/ordering tests, bulk storage compatibility tests, and four independent process readers checking equal version/mirror pairs from one bulk snapshot. The rebuilt-source functional run passed **602 tests**. An actual original 0.9.0 extension also read newly written bulk KV values, a streamed overwrite, and vector data alongside the new column tables.

Mapping, bounded update, observable background failure, callback reentrancy, and Windows file-release regressions are covered in `test_kv_mapping.py`. `cargo test --lib` passes all 44 native tests, including the restored collection suite.

Use `python -m pytest -q -o addopts= --disable-warnings` for a basic functional run. Performance timings are deliberately not asserted as fixed thresholds in the new regression tests.
