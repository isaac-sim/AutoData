# AutoData Documentation — Developer Guide

Build the docs in a dedicated Python 3.12 virtual environment. They only require
`docs/requirements.txt`; do not install Isaac Sim, Isaac Lab, Arena, or AutoData into this environment.
The commands below run on the **host**. The same Sphinx build also works inside the dev container.

## Prerequisites

`python3.12` and `python3.12-venv` must be installed on the host:

```bash
sudo apt-get install -y python3.12 python3.12-venv
```

## First-time setup

From a fresh terminal outside the simulation conda environment, create the venv and install dependencies.
If a conda environment is active, run `conda deactivate` first. Use the host Python 3.12 installed above:

```bash
cd docs
/usr/bin/python3.12 -m venv venv_docs
source venv_docs/bin/activate
python -m pip install -r requirements.txt
```

## Build and view (current branch/changes)

```bash
make html SPHINXOPTS="-W --keep-going"
xdg-open _build/current/html/index.html
```

For an existing docs environment, start at `cd docs` and `source venv_docs/bin/activate`, then run
the build command above. `-W` makes warnings fail the build. To rebuild every page after changing
the configuration, use `make html SPHINXOPTS="-E -a -W --keep-going"`.

If Python reports `Error in sitecustomize: No module named 'autodata_utils'`, an older editable
AutoData installation is exposing its startup hook without the renamed packages. The current
checkout tolerates that state; using a standalone docs environment also avoids loading the
simulation environment. No simulator reinstall is needed to build these docs.

## Multi-version docs

Builds docs for committed branches only (e.g. `main`, `release`). Local uncommitted changes are **not** reflected.

```bash
make multi-docs SPHINXOPTS="-W --keep-going"
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
