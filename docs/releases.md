# Stable and nightly builds

The tracked package version is read from `Cargo.toml`; `pyproject.toml` and Python
`__version__` derive it rather than maintaining independent version strings.
A tracked `1.0.0` does not itself publish a stable release.

## Nightly

The Nightly workflow follows KohakuTerrarium's split: it decides whether to build
and dispatches Release, which owns the single desktop/Android/sdist matrix and
existing PyPI trusted-publisher identity. It runs daily at 03:00 UTC or manually:

```bash
gh workflow run nightly.yml --ref main
# Rebuild a previously published source commit:
gh workflow run nightly.yml --ref main -f force_rebuild=true
```

Nightly requires successful push CI for the exact main commit. It skips an
already published commit unless forced. The shared Release workflow checks CI
again, pins every checkout to that SHA, and computes one version for all jobs.
Only build checkouts are stamped: Cargo `1.0.0-dev.YYYYMMDDHHMMSS` becomes Python
`1.0.0.devYYYYMMDDHHMMSS`. No version-bump commit or `v1.0.0` tag is created.

After all builds and native-host wheel smoke tests succeed, the workflow uploads
the wheels/sdist and a SHA256 manifest to a dated `nightly-YYYYMMDD` GitHub
prerelease (never Latest), then publishes the development version to PyPI.
Like the sibling project's procedure, the dated nightly release may create its
own non-version tag; it does not push a stable/version tag. The release notes
record the source SHA. The Nightly dispatcher finishing is not proof of shipping:
check the dispatched **Release** run and its publishing job too.

Install a specific nightly for reproducibility:

```bash
python -m pip install 'kohakuvault==1.0.0.devYYYYMMDDHHMMSS'
# Or opt into available pre-releases:
python -m pip install --pre --upgrade kohakuvault
```

Nightlies are not stable releases. Read [breaking changes](changes.md) before
opening existing data, especially legacy column databases. GitHub artifact
checksums validate downloaded bytes; they do not replace backups or testing.

## Stable

A `v*` version tag or published versioned GitHub release invokes the stable
publisher. Do not create either until the release is approved. Manual Release
with `channel=build` builds artifacts only; `channel=nightly` can publish only a
development version. Nightly GitHub release events are excluded from the stable
publisher, avoiding accidental publication of the tracked stable base version.

All releases retain existing SQLite durability defaults. CI tests installed
wheels on Linux, Windows and macOS across Python 3.10–3.14; release smoke tests
also exercise KV and column examples from installed native-host wheels. Android
and Linux ARM cross-built wheels are build-validated, not runtime-tested by the
native-host smoke step.
