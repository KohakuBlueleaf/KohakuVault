# Release Guide

This document explains how to set up GitHub Actions for automated builds and PyPI publishing.

## Prerequisites

### 1. PyPI Account Setup

1. **Create PyPI account**: https://pypi.org/account/register/
2. **Create Test PyPI account** (optional, for testing): https://test.pypi.org/account/register/

### 2. Generate API Tokens

#### Option A: API Token Authentication (Simpler)

**PyPI:**
1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Name: `github-actions-kohakuvault`
4. Scope: "Entire account" (or specific to "kohakuvault" project after first upload)
5. Copy the token (starts with `pypi-`)

**Test PyPI** (optional):
1. Go to https://test.pypi.org/manage/account/token/
2. Same steps as above
3. Copy the token

#### Option B: Trusted Publishing (Recommended, More Secure)

This method doesn't require storing tokens as secrets. Instead, PyPI trusts GitHub Actions directly.

1. First, manually upload your package once using API token (see "First Release" below)
2. Go to your PyPI project: https://pypi.org/manage/project/kohakuvault/settings/publishing/
3. Add a new "trusted publisher":
   - **Owner**: Your GitHub username or org
   - **Repository name**: `kohakuvault`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`
4. Save

If using trusted publishing, comment out the `password` line in `.github/workflows/release.yml` and uncomment the trusted publishing comments.

### 3. Add Secrets to GitHub Repository

Go to your GitHub repository: **Settings → Secrets and variables → Actions → New repository secret**

Add the following secrets:

| Secret Name | Description | Where to Get It |
|------------|-------------|-----------------|
| `PYPI_API_TOKEN` | PyPI API token | https://pypi.org/manage/account/token/ |
| `TEST_PYPI_API_TOKEN` | Test PyPI API token (optional) | https://test.pypi.org/manage/account/token/ |

**Important**:
- Secret names are case-sensitive
- Paste the entire token including the `pypi-` prefix
- Never commit tokens to git!

## GitHub Actions Workflows

### CI Workflow (`.github/workflows/ci.yml`)

**Triggers:**
- Push to `main` branch
- Pull requests to `main`
- Manual trigger (workflow_dispatch)

**What it does:**
- Tests on Python 3.10, 3.11, 3.12, 3.13
- Tests on Linux, Windows, macOS
- Runs pytest
- Checks Python formatting (black)
- Checks Rust formatting (rustfmt)

### Release Workflow (`.github/workflows/release.yml`)

**Triggers:**
- GitHub Release created
- Push tags matching `v*` (e.g., `v0.1.0`)
- Manual trigger (workflow_dispatch)

**What it does:**
1. **Build wheels**:
   - Python 3.10, 3.11, 3.12, 3.13
   - Linux (x86_64)
   - Windows (x86_64)
   - macOS (universal2 - Intel + Apple Silicon)
2. **Build sdist**: Source distribution
3. **Publish to PyPI**: Only on tags/releases
4. **Publish to Test PyPI**: Only on manual trigger

## Release Process

### First Release (Manual)

Before GitHub Actions can automatically publish, you need to create the project on PyPI:

```bash
# 1. Build the package locally
maturin build --release

# 2. Install twine
pip install twine

# 3. Upload to PyPI (first time)
twine upload target/wheels/kohakuvault-0.1.0-*.whl
# Enter your PyPI username and password when prompted
```

After this, GitHub Actions can publish updates automatically.

### Automated Release

1. **Update version** in `pyproject.toml` and `Cargo.toml`:
   ```toml
   # pyproject.toml
   version = "0.2.0"

   # Cargo.toml
   version = "0.2.0"
   ```

2. **Commit changes**:
   ```bash
   git add pyproject.toml Cargo.toml
   git commit -m "Bump version to 0.2.0"
   git push origin main
   ```

3. **Create a tag**:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. **GitHub Actions will automatically**:
   - Build wheels for all platforms and Python versions
   - Build source distribution
   - Publish to PyPI

5. **Alternatively, create a GitHub Release**:
   - Go to: https://github.com/yourusername/kohakuvault/releases/new
   - Tag: `v0.2.0`
   - Title: `Release 0.2.0`
   - Description: Changelog
   - Click "Publish release"
   - GitHub Actions will trigger automatically

### Testing Before Release

To test the build process without publishing to PyPI:

```bash
# Trigger the workflow manually
# Go to: Actions → Release → Run workflow
# This will upload to Test PyPI instead of PyPI
```

Or test locally:

```bash
# Build wheels locally
maturin build --release --interpreter python3.10 python3.11 python3.12 python3.13

# Check the wheels
ls -lh target/wheels/

# Test install
pip install target/wheels/kohakuvault-0.1.0-*.whl
```

## Environment Variables

The GitHub Actions workflows use these environment variables:

| Variable | Set By | Purpose |
|----------|--------|---------|
| `PYPI_API_TOKEN` | Repository Secret | PyPI authentication |
| `TEST_PYPI_API_TOKEN` | Repository Secret | Test PyPI authentication |
| `RUSTFLAGS` | Workflow (Linux only) | Enable CPU optimizations for Rust |

**Note**: `RUSTFLAGS: "-C target-cpu=native"` is only used on Linux builds to enable CPU-specific optimizations.

## Troubleshooting

### "Invalid API token" error

- Double-check the token is copied correctly (including `pypi-` prefix)
- Ensure the secret name matches exactly: `PYPI_API_TOKEN`
- Verify the token hasn't expired
- Check token scope covers the project

### Workflow doesn't trigger

- Ensure the tag starts with `v` (e.g., `v0.1.0`)
- Check the workflow file syntax is valid
- Verify workflows are enabled in repository settings

### Build fails on specific platform

- Check the build logs in GitHub Actions
- Test locally with: `maturin build --release`
- Ensure Rust code is compatible with all platforms

### macOS universal2 build fails

- Requires Rust 1.64+ with universal2 target
- The workflow automatically handles this

## Best Practices

1. **Test before tagging**: Always run tests locally before creating a release tag
2. **Semantic versioning**: Follow semver (MAJOR.MINOR.PATCH)
3. **Changelog**: Keep a changelog in the GitHub release notes
4. **Test PyPI**: Use Test PyPI for testing before production releases
5. **Trusted Publishing**: Switch to trusted publishing after first release for better security

## Resources

- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [Maturin Documentation](https://www.maturin.rs/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Semantic Versioning](https://semver.org/)
