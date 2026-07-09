# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Small source-read macros for the Isaac AutoData docs.

Mirrors the Arena docs tooling (``isaaclab_arena_doc_tools``): text substitutions applied to every
RST page before parsing, so recurring snippets (docker commands, code links, clone URLs) are defined
in one place and can switch between internal/external targets on release.
"""

import re
from typing import Any

from sphinx.application import Sphinx


def isaac_autodata_git_clone_code_block(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces the :isaac_autodata_git_clone_code_block: directive with a code block.

    The output git clone command depends on whether we're in release or internal mode.
    """

    def replacer(_: Any) -> str:
        config = app.config.isaac_autodata_docs_config
        git_clone_target = config["external_git_url"] if config["released"] else config["internal_git_url"]
        return f"""
.. code-block:: bash

    git clone --recurse-submodules {git_clone_target}

"""

    source[0] = re.sub(r":isaac_autodata_git_clone_code_block:", replacer, source[0])


def isaac_autodata_code_link(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces :isaac_autodata_code_link:`<relative/path>` with a link to the file on GitHub."""

    def replacer(match: re.Match) -> str:
        relative_path = match.group("relative_path")
        config = app.config.isaac_autodata_docs_config
        base_url = (
            config["external_code_link_base_url"] if config["released"] else config["internal_code_link_base_url"]
        )
        file_name = relative_path.split("/")[-1]
        return f"`{file_name} <{base_url}/{relative_path}>`_"

    source[0] = re.sub(r":isaac_autodata_code_link:`<(?P<relative_path>.*)>`", replacer, source[0])


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


def setup(app: Sphinx) -> None:
    app.connect("source-read", isaac_autodata_git_clone_code_block)
    app.connect("source-read", isaac_autodata_code_link)
    app.connect("source-read", docker_run_command_replacer)
    app.add_config_value("isaac_autodata_docs_config", {}, "env")
