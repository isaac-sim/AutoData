# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, OpenAI Codex, etc.) when working with code in this repository.

## Environment

All scripts and commands from this repo must be run inside a Docker container started with
`./docker/run_docker.sh` (or `./docker/run_docker.sh -c` for cuRobo). Users must start the container manually; if
no container started by the appropriate command exists, prompt the user to start it.

## Common Commands

### Linting, Formatting, and Coding Style

Pre-commit hooks enforce the style guide: black (line length 120), flake8, isort, pyupgrade (py312+), and codespell. Run checks manually **before** committing — not after:

```bash
# Run all checks (if hooks modify files, stage them and re-run before committing)
pre-commit run --all-files
```

### Coding Style

- Prefer `assert` over `if-then-raise ValueError` for internal invariant checks. Use `assert condition, "message"` instead of `if not condition: raise ValueError("message")`.
- Follow **PEP 8**. Method names and CLI arguments are `snake_case`.
- Use **modern type hints**: PEP 604 unions (`x | y`, `x | None`); do not use `typing.Union` or `typing.Optional`. Annotate concrete types on public interfaces (e.g. `torch.Tensor`, `wp.array(dtype=wp.vec3)`).
- Keep parameter names consistent across base/override APIs.
- **Name for autocomplete discoverability**:
  - Classes group by domain concept — `ActuatorNetLSTM`, not `LSTMActuatorNet`.
  - Methods put noun before modifier — `set_joint_position_target()`, not `set_target_joint_position()`.
- Prefer **nested classes** for helpers or enums only meaningful inside one parent class.
- Use **Google-style docstrings** on public APIs:
  - `Args:` entries use `name: description` (types live in annotations, not docstrings).
  - State SI units inline as `[unit]` for physical quantities (e.g. `positions [m]`, `[N, N·m]`). Skip non-physical fields (indices, counts, flags).
  - Use Sphinx roles (`:class:`, `:meth:`, `:attr:`) with short local references; avoid `_src` or private module paths.
- **File headers**: new files use the current-year SPDX copyright header; do not change the year on existing files.

## Guardrails (MUST FOLLOW, DO NOT DEVIATE FROM GUARDRAILS)

- Don't change or modify anything in the submodules. Restrict git related operations to read only, don't commit nor push any code. Don't create new branches in the submodules on your own.
- Don't mutate the `isaac_autodata` conda env. No `pip install/uninstall`, `conda install/remove`, or version upgrades. If a dependency is missing, surface the exact command and let the user run it.
- Don't re-run, modify, or work around `conda_installer.sh`. No env recreate/rename/delete, no channel or conda config changes. Installer issues get reported, not patched in-flight.
- Don't change any existing design docs or diagrams in /docs, that is only for human created files. If you need to add any new diagrams or designs, add them to the /agentic_design directory.

## Working on this codebase

Dev work is phased and coordinated via `docs/agents/`. Before starting any task:

1. Read `docs/agents/INDEX.md` — it names the active phase, sub-phase ownership, and reading order.
2. Load the sub-phase doc for the work you're doing, plus any contracts it depends on.
3. Any change that affects an interface signature or invariant must update its contract file in the same commit.
