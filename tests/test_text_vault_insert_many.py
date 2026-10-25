"""Tests for TextVault.insert_many - atomic batch inserts"""

import subprocess
import sys

import pytest

from kohakuvault import TextVault
from tests.test_kv_flush_isolation import _DEADLINE, _sql


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "batch.db")


def _row_counts(path, table="text_vault"):
    # Count from a subprocess so Python sqlite3 never shares the native writer's locks.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sqlite3, sys; c = sqlite3.connect(sys.argv[1]); t = sys.argv[2]; "
                "print(c.execute(f'SELECT count(*) FROM {t}').fetchone()[0], "
                "c.execute(f'SELECT count(*) FROM {t}_values').fetchone()[0]); c.close()"
            ),
            str(path),
            table,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=_DEADLINE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    docs, values = result.stdout.split()
    return int(docs), int(values)


def test_insert_many_returns_ids_in_order_and_matches_insert(db_path):
    tv = TextVault(db_path)
    tv.enable_auto_pack()
    single = tv.insert("alpha first document", {"n": 0})
    ids = tv.insert_many([("beta second document", {"n": 1}), ("gamma third document", [2, "x"])])
    assert ids == [single + 1, single + 2]
    assert tv.get_by_id(ids[0]) == ("beta second document", {"n": 1})
    assert tv.get_by_id(ids[1]) == ("gamma third document", [2, "x"])
    assert {hit[0] for hit in tv.search("document", k=10)} == {single, *ids}
    assert {hit[2]["n"] for hit in tv.search("second", k=10)} == {1}
    assert len(tv) == 3


def test_insert_many_accepts_any_iterable_and_empty(db_path):
    tv = TextVault(db_path)
    assert tv.insert_many([]) == []
    ids = tv.insert_many((f"doc {i}", f"v{i}".encode()) for i in range(5))
    assert len(ids) == 5 and len(set(ids)) == 5
    assert tv.get_by_id(ids[4]) == ("doc 4", b"v4")


def test_insert_many_multi_column(db_path):
    tv = TextVault(db_path, columns=["title", "body"])
    tv.enable_auto_pack()
    ids = tv.insert_many(
        [
            ({"title": "Intro", "body": "hello world"}, {"k": 1}),
            ({"title": "Outro", "body": "goodbye world"}, {"k": 2}),
        ]
    )
    assert tv.get_by_id(ids[1]) == ({"title": "Outro", "body": "goodbye world"}, {"k": 2})
    assert [hit[0] for hit in tv.search("Intro", column="title")] == [ids[0]]


@pytest.mark.parametrize(
    "bad",
    [
        ("missing value",),
        "not a pair",
        ({"wrong": "column"}, b"v"),
    ],
)
def test_invalid_item_inserts_nothing(db_path, bad):
    tv = TextVault(db_path)
    tv.insert("existing row", b"v")
    with pytest.raises(Exception):
        tv.insert_many([("valid row", b"v"), bad])
    assert _row_counts(db_path) == (1, 1)


def test_unpackable_value_inserts_nothing(db_path):
    tv = TextVault(db_path)
    tv.disable_auto_pack()
    with pytest.raises(Exception):
        tv.insert_many([("bytes row", b"ok"), ("object row", {"not": "bytes"})])
    assert _row_counts(db_path) == (0, 0)


def test_failed_write_rolls_back_the_batch(db_path):
    tv = TextVault(db_path)
    tv.insert("existing row", b"v")
    _sql(
        db_path,
        "CREATE TRIGGER reject AFTER INSERT ON text_vault_values "
        "WHEN (SELECT count(*) FROM text_vault_values) > 2 "
        "BEGIN SELECT RAISE(ABORT, 'rejected'); END;",
    )
    with pytest.raises(Exception, match="rejected"):
        tv.insert_many([("one", b"1"), ("two", b"2")])
    assert _row_counts(db_path) == (1, 1)
    assert tv.insert_many([("three", b"3")]) != []
    assert _row_counts(db_path) == (2, 2)
