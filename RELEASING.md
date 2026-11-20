# Releasing AirCompKit

## One-time setup (repository owner)

1. **Make the repository public** (Settings → General → Danger zone). Zenodo archives only public
   repositories.
2. **PyPI trusted publishing:** sign in at <https://pypi.org>, open *Your account → Publishing →
   Add a new pending publisher*, and enter: PyPI project `aircompkit`, owner `moyeendev`,
   repository `AirCompKit`, workflow `release.yml`, environment `pypi`. No API token is stored in
   the repository.
3. **GitHub environment:** Settings → Environments → New environment `pypi` (optionally with
   required reviewers, so every upload needs a click).
4. **Zenodo:** sign in at <https://zenodo.org> with GitHub, open *Account → GitHub* and switch
   the repository on. Every GitHub release is then archived with a DOI. `.zenodo.json` holds
   the metadata.
5. Check the authors in `pyproject.toml`, `CITATION.cff` and `.zenodo.json` (affiliations are optional there).

## Each release

1. Set `__version__` in `src/aircompkit/__init__.py` and `version` / `date-released` in
   `CITATION.cff`.
2. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` (the release notes
   are taken from this section) and start a new empty `[Unreleased]` section.
3. Check locally:

   ```bash
   pytest -q && ruff check src tests && mypy src
   python -m aircompkit reproduce --all --quick --check --data-root ./data
   python -m build && python -m twine check --strict dist/*
   ```

4. Commit, then tag and push:

   ```bash
   git tag -a vX.Y.Z -m "AirCompKit X.Y.Z"
   git push origin main vX.Y.Z
   ```

   `.github/workflows/release.yml` checks that the tag matches the version, builds, smoke-tests
   the wheel on Linux, Windows and macOS, uploads to PyPI and creates the GitHub release.
   Zenodo then mints the DOI.
5. Add the version DOI to `CITATION.cff` (`doi:`) and to the README, and use the concept DOI
   (all versions) in the paper's code-metadata table.
