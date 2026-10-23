"""Reproducible warm-process KV benchmark, including committed writes.

    python examples/benchmark_kv.py --structured --json results.json
    python examples/benchmark_kv.py --package-root target/performance-rework/legacy

Defaults: 10,000 keys, 256-byte raw values, median of three fresh-database trials.
Use --direct for slow scalar autocommit writes; --structured adds dict/list values.
Inputs, validation, open/close, and thread startup are outside operation timers.
This is not a cold-disk, power-loss durability, or concurrent-writer benchmark.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sqlite3
from statistics import median
import sys
import tempfile
import threading
from time import perf_counter


def timed(operation):
    start = perf_counter()
    value = operation()
    return perf_counter() - start, value


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fingerprint(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
    }


def benchmark(args, package, native):
    keys = [f"key:{index:010d}".encode() for index in range(args.keys)]
    # Distinct deterministic values detect mismatched keys, not just wrong lengths.
    raw = [
        (hashlib.sha256(key).digest() * ((args.value_bytes + 31) // 32))[: args.value_bytes]
        for key in keys
    ]
    datasets = [("raw", raw, args.keys * args.value_bytes)]
    if args.structured:
        structured = [
            {"id": index, "name": f"record-{index}", "scores": [index, index + 1], "ok": True}
            for index in range(args.keys)
        ]
        # A portable logical-size convention, NOT the native encoded storage size.
        logical_bytes = sum(
            len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
            for value in structured
        )
        datasets.append(("structured", structured, logical_bytes))

    samples = {}
    skipped = {}
    database_settings = []
    checkpoints = []
    capabilities = {
        name: callable(getattr(package.KVault, name, None)) for name in ("put_many", "get_many")
    }

    def record(name, seconds, operations, useful_bytes):
        row = samples.setdefault(
            name, {"seconds_samples": [], "operations": operations, "useful_bytes": useful_bytes}
        )
        row["seconds_samples"].append(seconds)

    with tempfile.TemporaryDirectory(prefix="kohakuvault-kv-benchmark-") as directory:
        for kind, values, useful_bytes in datasets:
            pairs = list(zip(keys, values))
            expected_items = dict(pairs)
            key_bytes = sum(map(len, keys))
            cap_bytes = max(8 << 20, 4 * (useful_bytes + key_bytes + 128 * args.keys))
            modes = ["scalar_cache", "bulk_direct"]
            if args.direct:
                modes.append("scalar_direct")
            else:
                skipped[f"{kind}.scalar_direct"] = "opt in with --direct (one commit per key)"

            for mode in modes:
                name = f"{kind}.{mode}"
                if mode.startswith("bulk") and not capabilities["put_many"]:
                    skipped[name] = "public put_many unavailable in this package"
                    continue
                for trial in range(args.repeats):
                    path = Path(directory) / f"{kind}-{mode}-{trial}.db"
                    vault = package.KVault(str(path), enable_wal=True)
                    observer = sqlite3.connect(str(path))
                    try:
                        if kind == "raw":
                            vault.disable_auto_pack()
                        else:
                            vault.enable_auto_pack()
                        journal = observer.execute("PRAGMA journal_mode").fetchone()[0]
                        require(journal.lower() == "wal", f"expected WAL, got {journal}")
                        database_settings.append(
                            {
                                "case": name,
                                "trial": trial,
                                "journal_mode_observed": journal,
                                "checkpoint_connection_synchronous_observed": observer.execute(
                                    "PRAGMA synchronous"
                                ).fetchone()[0],
                                "page_size_observed": observer.execute(
                                    "PRAGMA page_size"
                                ).fetchone()[0],
                                "write_cache_cap_bytes": cap_bytes if mode.endswith("cache") else 0,
                            }
                        )
                        cached = mode.endswith("cache")
                        if cached:
                            vault.enable_cache(cap_bytes=cap_bytes, flush_threshold=cap_bytes)

                        def put():
                            if mode.startswith("bulk"):
                                vault.put_many(pairs)
                            else:
                                for key, value in pairs:
                                    vault[key] = value

                        if cached:
                            start = perf_counter()
                            put()
                            accepted = perf_counter()
                            vault.flush_cache()
                            committed = perf_counter()
                            record(name + ".accept", accepted - start, args.keys, useful_bytes)
                            record(name + ".flush", committed - accepted, args.keys, useful_bytes)
                            record(name + ".committed", committed - start, args.keys, useful_bytes)
                            vault.disable_cache()
                        else:
                            seconds, _ = timed(put)
                            record(name + ".committed", seconds, args.keys, useful_bytes)

                        # External checkpoint exposes busy/errors instead of legacy close's suppression.
                        seconds, status = timed(
                            lambda: observer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                        )
                        require(status[0] == 0, f"checkpoint busy: {status}")
                        record(name + ".checkpoint", seconds, 1, 0)
                        checkpoints.append({"case": name, "trial": trial, "status": list(status)})
                        count = observer.execute("SELECT count(*) FROM kv").fetchone()[0]
                        require(count == args.keys, f"committed row count: {count} != {args.keys}")

                        # Cache is disabled: validate all persisted decoded values and warm reads.
                        actual = [vault[key] for key in keys]
                        require(actual == values, f"{name}: persisted values differ")
                        del actual
                        if mode != "scalar_cache":
                            continue

                        seconds, actual = timed(lambda: [vault[key] for key in keys])
                        record(kind + ".point_gets", seconds, args.keys, useful_bytes)
                        require(actual == values, "point read mismatch")
                        del actual
                        if capabilities["get_many"]:
                            seconds, actual = timed(lambda: vault.get_many(keys))
                            record(kind + ".get_many", seconds, args.keys, useful_bytes)
                            require(list(actual) == values, "bulk read mismatch/order mismatch")
                            del actual
                        else:
                            skipped[kind + ".get_many"] = (
                                "public get_many unavailable in this package"
                            )

                        # Legacy items() silently caps at 10k and has no limit argument.
                        unlimited_items = "limit" in inspect.signature(vault.items).parameters
                        if args.keys <= 10_000 or unlimited_items:
                            seconds, actual = timed(lambda: list(vault.items()))
                            record(kind + ".items", seconds, args.keys, useful_bytes + key_bytes)
                            require(
                                len(actual) == args.keys and dict(actual) == expected_items,
                                "items mismatch/duplicate/missing key",
                            )
                            del actual
                        else:
                            skipped[kind + ".items"] = (
                                "legacy items() capped at 10k; use --keys 10000"
                            )

                        ready = threading.Barrier(5)
                        go = threading.Event()

                        def reader():
                            # Start all workers and prime each native reader before timing.
                            vault[keys[0]]
                            ready.wait(timeout=30)
                            go.wait()
                            return [vault[key] for key in keys]

                        with ThreadPoolExecutor(max_workers=4) as executor:
                            futures = [executor.submit(reader) for _ in range(4)]
                            try:
                                ready.wait(timeout=30)
                                start = perf_counter()
                                go.set()
                                actual = [future.result() for future in futures]
                                seconds = perf_counter() - start
                            finally:
                                go.set()
                        record(
                            kind + ".four_shared_readers", seconds, args.keys * 4, useful_bytes * 4
                        )
                        require(all(result == values for result in actual), "thread read mismatch")
                        del actual
                    finally:
                        observer.close()
                        vault.close()
                        del vault, observer
                        # Legacy close does not release its native connection until destruction.
                        gc.collect()

    for row in samples.values():
        seconds = median(row["seconds_samples"])
        row["seconds_median"] = seconds
        row["operations_per_second"] = row["operations"] / seconds if seconds else None
        row["mib_per_second"] = row["useful_bytes"] / (1 << 20) / seconds if seconds else None

    return {
        "metadata": {
            "python": sys.version,
            "python_build": platform.python_build(),
            "python_compiler": platform.python_compiler(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "package_path": str(Path(package.__file__).resolve()),
            "package_version": package.__version__,
            "package_version_note": "May come from installed distribution metadata; paths/hashes identify code.",
            "proxy": fingerprint(Path(package.__file__).with_name("proxy.py")),
            "native_extension": fingerprint(native.__file__),
            "build_label": args.build_label,
            "build_label_note": "Caller supplied, not inferred from extension; no build profile API.",
            "python_observer_sqlite_version": sqlite3.sqlite_version,
            "synchronous": "NORMAL (library default; writer PRAGMA not exposed by public API)",
            "synchronous_note": "Not queried from observer: synchronous is connection-local. No settings weakened.",
            "wal_autocheckpoint": "library default (1000 pages); automatic checkpoint work stays in write time",
            "capabilities": capabilities,
        },
        "workload": {
            "keys": args.keys,
            "raw_value_bytes": args.value_bytes,
            "repeats": args.repeats,
            "structured": args.structured,
            "direct": args.direct,
            "notes": [
                "Fresh temporary DB per write mode/trial; fixed case order; medians of warm-process trials.",
                "Inputs constructed outside timers; packing/decoding and result allocation inside timers.",
                "Cache setup/disable excluded; accept + final flush = committed; no background flushing.",
                "Useful bytes exclude keys except items; structured values use compact sorted JSON UTF-8 size, not stored size.",
                "Explicit TRUNCATE checkpoint on Python sqlite3 observer timed separately after writes; close excluded.",
                "Checkpoint observer synchronization is recorded separately (0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA), not writer policy.",
                "All contents validated outside timers; reads warmed, write-back cache disabled.",
                "Four shared-handle threads each read all keys; startup excluded, dispatch/join included. Not a scaling claim.",
            ],
        },
        "database_settings": database_settings,
        "checkpoints": checkpoints,
        "results": samples,
        "skipped": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=int, default=10_000)
    parser.add_argument("--value-bytes", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--structured", action="store_true")
    parser.add_argument(
        "--package-root", type=Path, help="directory containing preserved kohakuvault/"
    )
    parser.add_argument("--build-label", default="unspecified")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if min(args.keys, args.value_bytes, args.repeats) <= 0:
        parser.error("keys, value-bytes, and repeats must be positive")
    if args.package_root:
        root = args.package_root.resolve()
        if not (root / "kohakuvault" / "__init__.py").is_file():
            parser.error("package-root must contain kohakuvault/__init__.py")
        sys.path.insert(0, str(root))

    import kohakuvault
    import kohakuvault._kvault as native

    if args.package_root:
        require(
            Path(kohakuvault.__file__).resolve().is_relative_to(root)
            and Path(native.__file__).resolve().is_relative_to(root),
            "package/native import escaped requested package root",
        )
    result = benchmark(args, kohakuvault, native)
    print(json.dumps(result["metadata"], indent=2))
    print(f"\nMedian of {args.repeats}; useful bytes are logical payload, not disk traffic")
    for name, row in result["results"].items():
        print(
            f"{name:39s} {row['seconds_median'] * 1000:10.3f} ms"
            f"  {row['operations']:8d} ops  {row['useful_bytes']:10d} bytes"
            f"  {row['mib_per_second']:9.2f} MiB/s"
        )
    for name, reason in result["skipped"].items():
        print(f"SKIP {name}: {reason}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
