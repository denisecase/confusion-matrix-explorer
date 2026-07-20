# Changelog

<!-- markdownlint-disable MD024 -->

Changes to this project will be documented in this file.

The format follows **[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)**
and this project adheres to **[Semantic Versioning](https://semver.org/spec/v2.0.0.html)**.

---

## [Unreleased]

### Added

- (placeholder) Notes for the next release.

---

## [1.1.0] - 2026-07-20

### Added

- Added current Python repository scaffolding using `uvx dc-up`.
- Added `zensical.toml` for documentation configuration and navigation.
- Added dedicated Zensical continuous-integration and GitHub Pages deployment workflows.
- Added automated link checking with Lychee.
- Added repository-wide EditorConfig, Git attributes, Markdown linting, YAML linting, and VS Code settings.
- Added `py_src_chores_pyshiny.ps1` for easier command execution.
- Added project guidance files for AI-assisted development and repository maintenance.

### Changed

- Migrated the documentation site from MkDocs to Zensical.
- Updated the project to require Python 3.14.
- Updated development and documentation dependencies.
- Updated GitHub Actions and Dependabot configuration.
- Updated the pre-commit configuration and repository validation checks.
- Reorganized the application as the `confusion_matrix_explorer` source package.
- Updated the Shinylive app to use package-relative imports for browser execution.
- Updated the documentation and developer instructions for building and previewing the app.
- Rebuilt the Shinylive application with current Pyodide and package assets.
- Changed the GitHub Pages process to export the Shinylive app before building the Zensical site.

### Fixed

- Fixed Shinylive package imports that failed in the browser environment.
- Fixed stale and mismatched Shinylive assets that caused integrity-check failures.
- Fixed local WebAssembly preview by serving the generated static site with HTTP server.
- Fixed Pyright errors for pandas-derived slider values.
- Fixed documentation links and link-checker exclusions for generated site and Shinylive files.

### Removed

- Removed the obsolete `mkdocs.yml` configuration.
- Removed the previous MkDocs deployment workflow.
- Removed obsolete MkDocs dependencies and generated MkDocs assets.

---

## [1.0.0] - 2025-11-10

### Added

- **Initial release**
- Includes actions and hosted docs/ using MkDocs

---

## Notes on versioning and releases

- **SemVer policy**
  - **MAJOR** - breaking API/schema or CLI changes.
  - **MINOR** - backward-compatible additions and enhancements.
  - **PATCH** - documentation, tooling, or non-breaking fixes.
- Versions are driven by git tags via `setuptools_scm`.
  Tag the repository with `vX.Y.Z` to publish a release.
- Documentation and badges are updated per tag and aliased to **latest**.

### Task 1. Update release metadata (manual edits)

1.1. `CITATION.cff` - update `version` and `date-released`
1.2. CHANGELOG.md: add section, move unreleased entries, update links
1.3. `pyproject.toml` - update build system `fallback-version`

### Task 2. Validate

Follow the commands in docs/DEVELOPER.md or if PowerShell, run:

```shell
.\py_src_chores_pyshiny.ps1
```

Make sure it works. Then hit CTRL c or CMD c (a couple times) to exit.

### Task 3. Commit, push, tag

```shell
git add -A
git commit -m "Prepare X.Y.Z"
git push -u origin main
```

Verify actions run on GitHub. After success:

```shell
git tag vX.Y.Z -m "X.Y.Z"
git push origin vX.Y.Z
```

## Only As Needed (delete a tag)

```shell
git tag -d vX.Z.Y
git push origin :refs/tags/vX.Z.Y
```

[Unreleased]: https://github.com/denisecase/confusion-matrix-explorer/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/denisecase/confusion-matrix-explorer/releases/tag/v1.1.0
[1.0.0]: https://github.com/denisecase/confusion-matrix-explorer/releases/tag/v1.0.0

<!-- markdownlint-enable MD024 -->
