# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable commands, runtime tabs and source-derived tables for the AutoData docs."""

import ast
import re
from pathlib import Path
from textwrap import indent
from typing import Any

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective


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


def _runtime_tabs(docker: str, uv: str) -> str:
    parts = [".. tab-set::\n   :sync-group: autodata-runtime\n"]
    for label, key, content in (("Docker", "docker", docker), ("Native uv", "uv", uv)):
        parts.append(f"   .. tab-item:: {label}\n      :sync: {key}\n\n{indent(content, '      ')}\n")
    return "\n".join(parts)


def _shell_block(command: str) -> str:
    return f".. code-block:: bash\n\n{indent(command, '   ')}"


class AutoDataPythonDirective(SphinxDirective):
    """Render one Python invocation's arguments for Docker and native uv."""

    has_content = True
    option_spec = {"no-sync": directives.flag}

    def run(self) -> list[nodes.Node]:
        self.assert_has_content()
        arguments = "\n".join(self.content)
        uv_mode = "--no-sync" if "no-sync" in self.options else "--locked"
        return self.parse_text_to_nodes(
            _runtime_tabs(_shell_block(f"python {arguments}"), _shell_block(f"uv run {uv_mode} python {arguments}"))
        )


class AutoDataRuntimeDirective(SphinxDirective):
    """Show where workflow commands run for each installation method."""

    option_spec = {"curobo": directives.flag}

    def run(self) -> list[nodes.Node]:
        curobo = "curobo" in self.options
        docker = (
            "From the repository root on the host, open or reattach to the development container:\n\n"
            + _shell_block("./docker/run_docker.sh" + (" -c" if curobo else ""))
            + "\n\nRun the following commands in its shell at ``/workspaces/autodata``."
        )
        installation = "installation-uv-curobo" if curobo else "installation-uv"
        uv = (
            f"Use a host terminal at the repository root with the :ref:`native uv environment <{installation}>`"
            + (" and cuRobo" if curobo else "")
            + " already installed. No container or environment activation is needed. "
            "The Native uv commands below select the project's environment."
        )
        return self.parse_text_to_nodes(_runtime_tabs(docker, uv))


def _declarations(app: Sphinx, decorator: str) -> list[tuple[str, dict[str, Any], str]]:
    """Name, options and summary line of every function in ``autodata_metrics/`` registered with ``decorator``.

    The declarations are parsed instead of imported, so the docs build needs none of the metrics' dependencies and
    the tables cannot drift from the code. In the summary, Sphinx roles become literals that need no reference
    target, and pipes are escaped so they do not read as substitutions.
    """
    found = []
    for path in sorted((Path(app.confdir).parent / "autodata_metrics").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.FunctionDef):
                continue
            for call in node.decorator_list:
                if isinstance(call, ast.Call) and getattr(call.func, "id", None) == decorator:
                    options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
                    summary = (ast.get_docstring(node) or "").split("\n")[0]
                    summary = re.sub(r":\w+:`~?([^`]+)`", r"``\1``", summary).replace("|", "\\|")
                    found.append((ast.literal_eval(call.args[0]), options, summary))
    return found


def _list_table(widths: str, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    lines = [".. list-table::", "   :header-rows: 1", f"   :widths: {widths}", ""]
    lines += ["   * - " + "\n     - ".join(row) for row in [header, *rows]]
    return "\n".join(lines) + "\n"


def autodata_metrics_table(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces ``:autodata_metrics_table:`` with a table of the built-in metrics."""
    if ":autodata_metrics_table:" not in source[0]:
        return
    rows = [
        (
            f"``{name}``",
            options["area"],
            options["unit"].replace("|", "\\|"),
            ", ".join(f"``{key}``" for key in options.get("requires", ())) or "none",
            summary,
        )
        for name, options, summary in sorted(_declarations(app, "metric"), key=lambda item: (item[1]["area"], item[0]))
    ]
    table = _list_table("20 12 14 26 28", ("Metric", "Area", "Unit", "Requires", "Definition"), rows)
    source[0] = source[0].replace(":autodata_metrics_table:", table)


def autodata_checks_table(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces ``:autodata_checks_table:`` with a table of the episode checks."""
    if ":autodata_checks_table:" not in source[0]:
        return
    declarations = [
        (options.get("group", "correctness"), name, summary) for name, options, summary in _declarations(app, "check")
    ]
    rows = [(f"``{name}``", group, summary) for group, name, summary in sorted(declarations)]
    source[0] = source[0].replace(
        ":autodata_checks_table:", _list_table("22 14 64", ("Check", "Group", "Definition"), rows)
    )


def setup(app: Sphinx) -> None:
    app.add_js_file("runtime-tabs.js")
    app.add_directive("autodata-python", AutoDataPythonDirective)
    app.add_directive("autodata-runtime", AutoDataRuntimeDirective)
    app.connect("source-read", autodata_git_clone_code_block)
    app.connect("source-read", autodata_code_link)
    app.connect("source-read", docker_run_command_replacer)
    app.connect("source-read", autodata_metrics_table)
    app.connect("source-read", autodata_checks_table)
    app.add_config_value("autodata_docs_config", {}, "env")
