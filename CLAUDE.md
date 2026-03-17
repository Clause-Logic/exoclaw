# exoclaw

## Releasing

Releases are triggered by pushing a git tag. A GitHub Actions workflow (`.github/workflows/release.yml`) runs automatically and creates the GitHub release.

```bash
git tag v0.6.0
git push origin v0.6.0
```

Do not create GitHub releases manually — let the workflow handle it.
