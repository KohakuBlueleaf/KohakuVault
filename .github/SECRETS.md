# GitHub Actions Secrets Setup

This is a quick reference for setting up the required secrets for automated PyPI publishing.

## Required Secrets

Go to: **GitHub Repository → Settings → Secrets and variables → Actions → New repository secret**

### 1. PYPI_API_TOKEN

**Description**: PyPI API token for publishing releases

**How to get it**:
1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Name: `github-actions-kohakuvault`
4. Scope: "Entire account" (initially) or "Project: kohakuvault" (after first upload)
5. Click "Create token"
6. **Copy the token** (starts with `pypi-`) - you won't see it again!

**Add to GitHub**:
- Secret name: `PYPI_API_TOKEN`
- Secret value: Paste the entire token (including `pypi-` prefix)

### 2. TEST_PYPI_API_TOKEN (Optional)

**Description**: Test PyPI API token for testing releases before production

**How to get it**:
1. Go to https://test.pypi.org/manage/account/token/
2. Follow same steps as above
3. Copy the token

**Add to GitHub**:
- Secret name: `TEST_PYPI_API_TOKEN`
- Secret value: Paste the entire token

## Alternative: Trusted Publishing (Recommended)

Instead of storing API tokens, you can use PyPI's trusted publishing feature:

**Advantages**:
- No secrets to manage
- More secure (GitHub OIDC authentication)
- No token expiration issues

**Setup**:
1. First, do one manual release with an API token
2. Go to https://pypi.org/manage/project/kohakuvault/settings/publishing/
3. Add trusted publisher:
   - **Owner**: Your GitHub username/org
   - **Repository**: `kohakuvault`
   - **Workflow**: `release.yml`
   - **Environment**: `pypi`
4. Update `.github/workflows/release.yml`:
   - Comment out the `password: ${{ secrets.PYPI_API_TOKEN }}` line
   - The workflow already has `id-token: write` permission needed

## Verifying Setup

After adding secrets:

1. **Check secrets are added**:
   - Go to: Settings → Secrets and variables → Actions
   - You should see: `PYPI_API_TOKEN` (and optionally `TEST_PYPI_API_TOKEN`)
   - You can't view the value, but you can update/delete them

2. **Test the workflow**:
   ```bash
   # Option 1: Manual trigger (publishes to Test PyPI)
   # Go to: Actions → Release → Run workflow

   # Option 2: Create a test tag
   git tag v0.0.1-test
   git push origin v0.0.1-test
   # Check Actions tab for build progress
   ```

## Security Best Practices

1. **Never commit tokens** to git
2. **Use organization secrets** for multi-repo setups
3. **Scope tokens narrowly** - limit to specific projects when possible
4. **Rotate tokens periodically** - regenerate every 6-12 months
5. **Use trusted publishing** when possible - eliminates token management
6. **Review workflow logs** - ensure tokens aren't accidentally printed

## Troubleshooting

### Secret not working?

- ✅ Check secret name is exactly `PYPI_API_TOKEN` (case-sensitive)
- ✅ Ensure you copied the entire token including `pypi-` prefix
- ✅ Verify the token hasn't expired
- ✅ Check token scope covers your project
- ✅ Make sure you saved the secret (not just previewed)

### Workflow can't access secret?

- Secrets are not available in forked repositories (security feature)
- Only repository maintainers can add secrets
- Check workflow uses correct secret name: `${{ secrets.PYPI_API_TOKEN }}`

### Want to switch to trusted publishing?

1. Remove `PYPI_API_TOKEN` secret (or leave it as backup)
2. Configure trusted publisher on PyPI
3. Update workflow file (comment out password line)
4. Test with a new release

## Resources

- [PyPI API Tokens](https://pypi.org/help/#apitoken)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [GitHub Encrypted Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
