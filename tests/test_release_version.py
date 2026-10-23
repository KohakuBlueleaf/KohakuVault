"""Nightly version stamping must never mutate dependency versions."""

from datetime import datetime, timezone
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "release_version", Path(__file__).parents[1] / "scripts/release_version.py"
)
_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_version)


def test_nightly_version_is_pep440_prerelease():
    from packaging.version import Version

    text = '[package]\nname = "kohakuvault"\nversion = "1.0.0"\n[dependencies]\nversion = "0.9.0"\n'
    result = _version.resolve(text, True, datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
    assert result["cargo_version"] == "1.0.0-dev.20260102030405"
    assert result["python_version"] == "1.0.0.dev20260102030405"
    assert Version(result["python_version"]) < Version("1.0.0")
    assert result["tag"] == "nightly-20260102"
    stamped = _version.stamp(text, result["cargo_version"])
    assert '[dependencies]\nversion = "0.9.0"' in stamped


def test_stable_resolution_and_invalid_stamp():
    result = _version.resolve('[package]\nversion = "1.0.0"\n')
    assert result["tag"] == "v1.0.0"
    assert result["nightly"] == "false"
    with pytest.raises(ValueError):
        _version.stamp('[package]\nversion = "1.0.0"\n', "1.0.0; echo no")
    with pytest.raises(ValueError):
        _version.resolve('[package]\nversion = "1.0.0-dev.1"\n')
