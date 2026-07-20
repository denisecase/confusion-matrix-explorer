#Requires -Version 7.0

<#
============================================================
py_src_chores.ps1 (CONFUSION-MATRIX-EXPLORER ONLY)
============================================================
Updated: 2026-07-20

Update dependencies, lint, test, and build docs.
For Python source repos only.

Run with:
.\py_src_chores.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

uv self update
uv python pin 3.14
uv lock --upgrade
uv sync --extra dev --extra docs --upgrade

uvx pre-commit install
uvx pre-commit autoupdate

git add -A
uvx pre-commit run --all-files
# Repeat because hooks may modify files.
uvx pre-commit run --all-files

# run common chores
uv run ruff check . --fix
uv run ruff format .

uv run python -m pyright
uv run python -m pytest
uv run python -m zensical build

## 1. Delete any existing `docs/app` directory.
uv run python -c "import shutil; shutil.rmtree('docs/app', ignore_errors=True)"

## 2. Export the Shinylive app to `docs/app`.
uv run shinylive export ./src/confusion_matrix_explorer ./docs/app

## 3. Build the complete documentation site.
uv run zensical build

Write-Host "All commands executed successfully."
Write-Host "Run a Python module to verify .venv/ is working correctly."
Write-Host "Starting a local server to preview the app at http://127.0.0.1:8008..."

## 4. Preview the documentation site locally.
uv run python -m http.server 8008 --bind 127.0.0.1 --directory docs/app
