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
uv python pin 3.12
uv venv

.venv\Scripts\activate # Windows
# source .venv/bin/activate  # Mac/Linux/WSL

uv sync --extra dev --extra docs --upgrade
uv run pre-commit install
uv run shiny run --reload src/confusion_matrix_explorer/app.py
```

## Dev 2. Validate Local Changes

```shell
git pull origin main
uvx pre-commit autoupdate
git add .
uvx ruff check . --fix
uvx ruff format .
uvx deptry .
uv run pytest
```

Run the pre-commit hooks (twice, if needed):

```shell
pre-commit run --all-files
```

## DEV 3. Build and Preview The App and Documentation

Use the commands below to build and copy the app to the docs/app folder for deploying via GitHub Pages. 
Note: Building the app takes a lot of space (400 MB) in addition to the .venv install. 

```shell
uv run shinylive export ./src/confusion_matrix_explorer ./docs/app
uv run python -m http.server --directory docs\app --bind localhost 8008 
uv run mkdocs build --strict
uv run mkdocs serve
```

Verify local API docs at: <http://localhost:8000>
When done reviewing, use CTRL c or CMD c to quit.

## DEV 4. Test

Update `CHANGELOG.md` and `pyproject`.toml dependencies.
Ensure CI passes.

```shell
git add .
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

Open the URL (usually http://127.0.0.1:8008) to verify.

Once hosted:

- App: <https://denisecase.github.io/confusion-matrix-explorer/app/>
- Docs: <https://denisecase.github.io/confusion-matrix-explorer/>
