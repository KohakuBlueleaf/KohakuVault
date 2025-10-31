# GitHub Actions Workflows Explained

This document explains how the CI/CD workflows are structured and why.

## CI Workflow (`.github/workflows/ci.yml`)

### Why `maturin build` instead of `maturin develop`?

**Problem**: GitHub Actions runners don't use virtual environments by default. When you run `maturin develop`, it requires a virtualenv and will fail with:

```
💥 maturin failed
  Caused by: Couldn't find a virtualenv or conda environment, but you need one to use this command.
```

**Solution**: Use `maturin build` + `pip install` instead:

```yaml
- name: Build wheel
  run: maturin build --release

- name: Install built wheel
  run: pip install target/wheels/*.whl
```

This approach:
- ✅ Works without a virtualenv
- ✅ Tests the actual wheel that users will install
- ✅ Faster and more reliable in CI
- ✅ Standard practice for Rust+Python projects

### Workflow Steps Explained

1. **Checkout code**: Clone the repository
2. **Set up Python**: Install specific Python version (3.10, 3.11, 3.12, 3.13)
3. **Set up Rust**: Install Rust toolchain
4. **Cache Rust**: Cache Rust build artifacts for faster builds
5. **Install maturin**: Install the build tool
6. **Build wheel**: Compile Rust extension into a wheel
7. **Install wheel**: Install the built wheel + test dependencies
8. **Run tests**: Execute pytest
9. **Format checks**: Check code formatting (only on one matrix job)

### Matrix Strategy

Tests run on all combinations of:
- **OS**: Ubuntu (Linux), Windows, macOS
- **Python**: 3.10, 3.11, 3.12, 3.13

Total: **12 test jobs** (3 OS × 4 Python versions)

## Release Workflow (`.github/workflows/release.yml`)

### Why separate jobs?

The workflow is split into multiple jobs for efficiency and clarity:

1. **build-wheels**: Builds wheels for each platform (parallel)
2. **build-sdist**: Builds source distribution
3. **publish-to-pypi**: Publishes to production PyPI (only on tags/releases)
4. **publish-to-test-pypi**: Publishes to Test PyPI (only on manual trigger)

### Platform-Specific Builds

**Linux**:
```yaml
maturin build --release --interpreter python3.10 python3.11 python3.12 python3.13
```
- Builds manylinux wheels (compatible with most Linux distros)
- Uses standard x86_64 architecture

**Windows**:
```yaml
maturin build --release --interpreter python3.10 python3.11 python3.12 python3.13
```
- Builds standard Windows x86_64 wheels

**macOS**:
```yaml
maturin build --release --interpreter python3.10 python3.11 python3.12 python3.13 --target universal2-apple-darwin
```
- Builds **universal2** wheels (Intel + Apple Silicon in one wheel)
- Works on both Intel Macs and M1/M2/M3 Macs
- Larger file size but better compatibility

### Artifacts

Each platform job uploads its wheels as artifacts:
- `wheels-ubuntu-latest`: Linux wheels
- `wheels-windows-latest`: Windows wheels
- `wheels-macos-latest`: macOS universal2 wheels
- `sdist`: Source distribution

The publish job downloads all artifacts and uploads them to PyPI in one go.

### Trigger Conditions

**When does the workflow run?**

1. **Git tag starting with `v`** (e.g., `v0.1.0`):
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
   → Builds wheels and publishes to PyPI

2. **GitHub Release created**:
   - Create release on GitHub UI
   → Builds wheels and publishes to PyPI

3. **Manual trigger** (workflow_dispatch):
   - Go to Actions → Release → Run workflow
   → Builds wheels and publishes to **Test PyPI** (for testing)

**Publishing conditions**:
- PyPI: Only when triggered by tag or release
- Test PyPI: Only when manually triggered

This prevents accidental publishes!

## Environment Variables

### Set by Workflow

| Variable | Where | Value | Purpose |
|----------|-------|-------|---------|
| `RUSTFLAGS` | Linux build | `-C target-cpu=native` | Enable CPU optimizations |

Note: `target-cpu=native` is only safe for Linux because GitHub Actions uses consistent runners. Not used on Windows/macOS to ensure broader compatibility.

### User-Provided Secrets

| Secret | Required | Purpose |
|--------|----------|---------|
| `PYPI_API_TOKEN` | ✅ Yes | Authenticate to PyPI |
| `TEST_PYPI_API_TOKEN` | ⚠️ Optional | Authenticate to Test PyPI |

See [SECRETS.md](SECRETS.md) for setup instructions.

## Testing Locally

### Test CI workflow locally:

```bash
# Install maturin
pip install maturin

# Build wheel
maturin build --release

# Install and test
pip install target/wheels/*.whl
pytest
```

### Test release build:

```bash
# Build for multiple Python versions
maturin build --release --interpreter python3.10 python3.11 python3.12 python3.13

# Check wheels
ls -lh target/wheels/

# Test install
pip install target/wheels/kohakuvault-0.1.0-*.whl
python -c "from kohakuvault import KVault; print('Success!')"
```

### Test macOS universal2 locally (macOS only):

```bash
rustup target add aarch64-apple-darwin x86_64-apple-darwin
maturin build --release --target universal2-apple-darwin
```

## Troubleshooting

### CI fails with "Couldn't find a virtualenv"

- ✅ **Fixed**: Use `maturin build` instead of `maturin develop`
- The current workflow already uses this approach

### Wheel doesn't work on specific Python version

- Check that the Python version is in the interpreter list
- Verify Python version is supported in `pyproject.toml`

### macOS universal2 build fails

- Requires Rust 1.64+ with universal2 support
- GitHub Actions runners are usually up-to-date

### PyPI publish fails with "Invalid token"

- Check `PYPI_API_TOKEN` secret is set correctly
- Ensure token hasn't expired
- Verify token scope includes the project

### Want to test without publishing?

- Use manual trigger (publishes to Test PyPI)
- Or comment out the publish jobs temporarily

## Best Practices

1. **Always test locally first**: Run `maturin build` and test the wheel before pushing tags
2. **Use semantic versioning**: MAJOR.MINOR.PATCH (e.g., 0.1.0, 0.2.0, 1.0.0)
3. **Write release notes**: Document changes in GitHub releases
4. **Monitor CI**: Check that tests pass before tagging
5. **Test PyPI first**: Use manual trigger to test publishing flow

## Resources

- [Maturin Documentation](https://www.maturin.rs/)
- [GitHub Actions - Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
- [PyO3 Guide](https://pyo3.rs/)
- [PyPI Publishing Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
