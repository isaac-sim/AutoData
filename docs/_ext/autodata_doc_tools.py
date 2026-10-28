# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small source-read macros for the AutoData docs.

Mirrors the Arena docs tooling (``isaaclab_arena_doc_tools``): text substitutions applied to every
RST page before parsing, so recurring snippets (docker commands, code links, clone URLs) are defined
in one place and can switch between internal/external targets on release.
"""

import ast
import re
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx


def autodata_git_clone_code_block(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces the :autodata_git_clone_code_block: directive with a code block.

    The output git clone command depends on whether we're in release or internal mode.
    """

    def replacer(_: Any) -> str:
        config = app.config.autodata_docs_config
        git_clone_target = config["external_git_url"] if config["released"] else config["internal_git_url"]
        return f"""
.. code-block:: bash

    git clone --recurse-submodules {git_clone_target}

"""

    source[0] = re.sub(r":autodata_git_clone_code_block:", replacer, source[0])


def autodata_code_link(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces :autodata_code_link:`<relative/path>` with a link to the file on GitHub."""

    def replacer(match: re.Match) -> str:
        relative_path = match.group("relative_path")
        config = app.config.autodata_docs_config
        base_url = (
            config["external_code_link_base_url"] if config["released"] else config["internal_code_link_base_url"]
        )
        file_name = relative_path.split("/")[-1]
        return f"`{file_name} <{base_url}/{relative_path}>`_"

    source[0] = re.sub(r":autodata_code_link:`<(?P<relative_path>.*)>`", replacer, source[0])


def docker_run_command_replacer(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces docker run command directives with code blocks.

    * ``:docker_run_default:`` -- the base dev container.
    * ``:docker_run_curobo:`` -- the cuRobo/SkillGen image (``-c``).
    """

    def default_replacer(_: Any) -> str:
        return """.. code-block:: bash

           ./docker/run_docker.sh"""

    def curobo_replacer(_: Any) -> str:
        return """.. code-block:: bash

           ./docker/run_docker.sh -c"""

    source[0] = re.sub(r":docker_run_default:", default_replacer, source[0])
    source[0] = re.sub(r":docker_run_curobo:", curobo_replacer, source[0])


def _first_line(docstring: str | None) -> str:
    """The docstring's summary line, with Sphinx roles turned into literals that need no reference target."""
    return re.sub(r":\w+:`~?([^`]+)`", r"``\1``", (docstring or "").split("\n")[0])


def autodata_metrics_table(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces ``:autodata_metrics_table:`` with a table of the built-in metrics.

    The rows are read from the ``@metric`` declarations in ``autodata_metrics/`` instead of importing them, so the
    docs build needs none of the metrics' dependencies and the table cannot drift from the code.
    """
    if ":autodata_metrics_table:" not in source[0]:
        return
    rows = []
    for path in sorted((Path(app.confdir).parent / "autodata_metrics").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and getattr(decorator.func, "id", None) == "metric":
                    options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in decorator.keywords}
                    requires = ", ".join(f"``{key}``" for key in options.get("requires", ())) or "none"
                    definition = _first_line(ast.get_docstring(node))
                    name = ast.literal_eval(decorator.args[0])
                    rows.append((options["area"], name, options["unit"], requires, definition))
    table = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 20 12 14 26 28",
        "",
        "   * - Metric\n     - Area\n     - Unit\n     - Requires\n     - Definition",
    ]
    for area, name, unit, requires, definition in sorted(rows):
        unit, definition = (text.replace("|", "\\|") for text in (unit, definition))
        table.append(f"   * - ``{name}``\n     - {area}\n     - {unit}\n     - {requires}\n     - {definition}")
    source[0] = source[0].replace(":autodata_metrics_table:", "\n".join(table) + "\n")


def autodata_checks_table(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces ``:autodata_checks_table:`` with a table of the episode checks, read like the metrics table."""
    if ":autodata_checks_table:" not in source[0]:
        return
    rows = []
    for path in sorted((Path(app.confdir).parent / "autodata_metrics").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and getattr(decorator.func, "id", None) == "check":
                    options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in decorator.keywords}
                    definition = _first_line(ast.get_docstring(node))
                    rows.append((options.get("group", "correctness"), ast.literal_eval(decorator.args[0]), definition))
    table = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 22 14 64",
        "",
        "   * - Check\n     - Group\n     - Definition",
    ]
    for group, name, definition in sorted(rows):
        table.append(f"   * - ``{name}``\n     - {group}\n     - {definition}")
    source[0] = source[0].replace(":autodata_checks_table:", "\n".join(table) + "\n")


def setup(app: Sphinx) -> None:
    app.connect("source-read", autodata_git_clone_code_block)
    app.connect("source-read", autodata_code_link)
    app.connect("source-read", docker_run_command_replacer)
    app.connect("source-read", autodata_metrics_table)
    app.connect("source-read", autodata_checks_table)
    app.add_config_value("autodata_docs_config", {}, "env")
