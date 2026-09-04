# Autodata Documentation — Developer Guide

The docs are built on the **host machine** (not inside Docker) using a dedicated Python 3.12 venv.

## Prerequisites

`python3.12` and `python3.12-venv` must be installed on the host:

```bash
sudo apt-get install -y python3.12 python3.12-venv
```

## First-time setup

From the repo root, create the venv and install dependencies:

```bash
cd docs
python3.12 -m venv venv_docs
source venv_docs/bin/activate
pip install -r requirements.txt
```

## Build and view (current branch/changes)

```bash
make html
xdg-open _build/current/html/index.html
```

## Multi-version docs

Builds docs for committed branches only (e.g. `main`, `release`). Local uncommitted changes are **not** reflected.

```bash
make multi-docs
xdg-open _build/index.html
```

## Writing conventions

- Pages live under `pages/<section>/`. Add new pages to a `toctree` in `index.rst` (or the
  section's own `index.rst`) or the build will warn about orphaned documents.
- Unfinished sections are marked with `.. todo::` directives. These render in the built HTML
  (see `todo_include_todos` in `conf.py`), so placeholders are visible while the docs are drafted.
- Macros available in any page (defined in `_ext/autodata_doc_tools.py`):
  - `:docker_run_default:` — inserts the base dev-container command.
  - `:docker_run_curobo:` — inserts the cuRobo/SkillGen container command.
  - `` :autodata_code_link:`<path/to/file.py>` `` — links to a file on GitHub.
  - `:autodata_git_clone_code_block:` — inserts the clone command.
- Images go in `images/` and are referenced with relative paths (see the Arena docs for examples).
