# DEVELOPER.md

## Prerequisites: Set Up Machine

- View hidden files and folders
- View file extensions
- Git
- VS Code (recommended)
- **[uv](https://github.com/astral-sh/uv)**

## Fork and Clone Repository

1. Fork the repo.
2. Clone your repo to your machine and open it in VS Code.

Open a terminal and run the following commands.

```shell
git clone https://github.com/YOUR_USERNAME/confusion-matrix-explorer.git
cd confusion-matrix-explorer
```

## Dev 1. One-time setup

- Open the repo directory in VS Code.
- Open a terminal in VS Code.

```shell
# if newly downloaded from GitHub you may need to give permission:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# then run a single PowerShell script that does the following:
.\py_src_chores_pyshiny.ps1
```

Alternatively, run the commands one at a time:

```shell
uv self update
uv python pin 3.14
uv lock --upgrade
uv sync --extra dev --extra docs --upgrade

uvx pre-commit install
uvx pre-commit autoupdate

git add -A
uvx pre-commit run --all-files
# rerun if changes
uvx pre-commit run --all-files

uv run shiny run --reload src/confusion_matrix_explorer/app.py
```

## Dev 2. Validate Local Changes

```shell
git pull origin main
git add .

uv run ruff check . --fix
uv run ruff format .
uv run pytest

git add -A
```

## DEV 3. Build and Preview The App and Documentation

Use the commands below to:

1. Delete any existing `docs/app` directory.
2. Export the Shinylive app to `docs/app`.
3. Build the complete documentation site.
4. Preview the documentation site locally.

Note: Building the app takes a lot of space (400 MB) in addition to the .venv install.

```shell
uv run python -c "import shutil; shutil.rmtree('docs/app', ignore_errors=True)"

uv run shinylive export ./src/confusion_matrix_explorer ./docs/app

uv run zensical build

uv run python -m http.server 8008 --bind 127.0.0.1 --directory docs/app
```

Verify local API docs at: <http://127.0.0.1:8008/>.

It should **Launch the Confusion Matrix Explorer**.

It may take a while to open (lots of code to make it work).

## Stop The App

When done reviewing, use **CTRL c** or **CMD c** (possibly a couple times) to quit.

## DEV 4. After Making Any Changes: Test

Update `CHANGELOG.md` and `pyproject`.toml dependencies.
Ensure CI passes.

```shell
git add -A
uv run pre-commit run --all-files
uv run pytest -q
```

## DEV 5. Git add-commit-push Changes

```shell
git add .
git commit -m "Prep vx.y.z"
git push -u origin main
```

## DEV 8. Git tag and Push tag

**Important:** Wait for GitHub Actions from prior step to complete successfully (all green checks).
If any fail, fix issues and push again before tagging.

```shell
git tag vx.y.z -m "x.y.z"
git push origin vx.y.z
```

## Building the ShinyLive Part for GitHub Pages

This lives in ./shinylive_app/.

1. Copy in utils_confusion.py
2. Copy in app.py
3. Edit app.py to use local imports from utils_confusion.py.
4. Export with shinylive using the command below.
5. Preview locally.

```shell
uv run shinylive export ./shinylive_app ./docs/app
uv run python -m http.server --directory docs/app --bind localhost 8008
```

Be patient, it may take a while to load.

Open the URL (usually <http://127.0.0.1:8008>) to verify.

Once hosted:

- App: <https://denisecase.github.io/confusion-matrix-explorer/app/>
- Docs: <https://denisecase.github.io/confusion-matrix-explorer/>
