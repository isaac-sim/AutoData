# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Strict, duplicate-safe YAML loading for autonomous configuration."""

from __future__ import annotations

import re
import yaml
from pathlib import Path
from typing import Any
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from isaac_autodata_interfaces.autonomous.errors import AutonomousValidationError, ValidationIssue

MAX_YAML_BYTES = 1_000_000
"""Maximum UTF-8 document size accepted by the offline compiler."""


class _TaskSafeLoader(yaml.SafeLoader):
    """Safe loader with YAML 1.2 boolean semantics.

    PyYAML's default YAML 1.1 resolver treats robotics relation names such as ``on`` and ``off``
    as booleans. Task request files use those words as Arena semantic strings, so only the YAML
    1.2 spellings ``true`` and ``false`` are resolved as booleans here.
    """


_TaskSafeLoader.yaml_implicit_resolvers = {
    key: list(resolvers) for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
for first_character, resolvers in _TaskSafeLoader.yaml_implicit_resolvers.items():
    _TaskSafeLoader.yaml_implicit_resolvers[first_character] = [
        (tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"
    ]
_TaskSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def load_yaml_document(path: str | Path) -> Any:
    """Load one safe YAML document while retaining duplicate-key detection and field paths.

    Args:
        path: YAML file to load.

    Returns:
        Nested Python scalars, lists, and dictionaries.

    Raises:
        AutonomousValidationError: If the file cannot be read, exceeds the size limit, contains
            invalid YAML, duplicate/complex mapping keys, recursive aliases, or multiple documents.
    """

    source_path = Path(path).expanduser()
    try:
        raw = source_path.read_bytes()
    except FileNotFoundError:
        raise AutonomousValidationError(
            [ValidationIssue((), "file_not_found", f"YAML file does not exist: {source_path}")]
        ) from None
    except OSError as exc:
        raise AutonomousValidationError(
            [ValidationIssue((), "file_read_error", f"could not read YAML file: {exc}")]
        ) from None

    if len(raw) > MAX_YAML_BYTES:
        raise AutonomousValidationError([
            ValidationIssue(
                (),
                "document_too_large",
                f"YAML document is {len(raw)} bytes; maximum is {MAX_YAML_BYTES}",
            )
        ])
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AutonomousValidationError(
            [ValidationIssue((), "invalid_encoding", f"YAML must be UTF-8: {exc}")]
        ) from None
    return _load_yaml_text(text)


def _load_yaml_text(text: str) -> Any:
    loader = _TaskSafeLoader(text)
    try:
        node = loader.get_single_node()
        if node is None:
            raise AutonomousValidationError([ValidationIssue((), "empty_document", "YAML document is empty")])
        return _construct_node(loader, node, (), set(), set())
    except AutonomousValidationError:
        raise
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = ""
        if mark is not None:
            location = f" at line {mark.line + 1}, column {mark.column + 1}"
        problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
        raise AutonomousValidationError(
            [ValidationIssue((), "yaml_syntax", f"invalid YAML{location}: {problem}")]
        ) from None
    finally:
        loader.dispose()


def _construct_node(
    loader: yaml.SafeLoader,
    node: Node,
    path: tuple[str | int, ...],
    active_nodes: set[int],
    seen_nodes: set[int],
) -> Any:
    node_id = id(node)
    if node_id in active_nodes:
        raise AutonomousValidationError(
            [ValidationIssue(path, "recursive_alias", "recursive YAML aliases are not allowed")]
        )
    if node_id in seen_nodes:
        raise AutonomousValidationError(
            [
                ValidationIssue(
                    path,
                    "yaml_alias_not_allowed",
                    "YAML aliases are not allowed in task requests",
                )
            ]
        )
    seen_nodes.add(node_id)
    active_nodes.add(node_id)
    try:
        if isinstance(node, MappingNode):
            result: dict[str, Any] = {}
            for key_node, value_node in node.value:
                if not isinstance(key_node, ScalarNode) or key_node.tag != "tag:yaml.org,2002:str":
                    raise AutonomousValidationError(
                        [ValidationIssue(path, "invalid_mapping_key", "mapping keys must be strings")]
                    )
                key = key_node.value
                key_path = path + (key,)
                if key in result:
                    raise AutonomousValidationError(
                        [ValidationIssue(key_path, "duplicate_key", f"mapping key {key!r} is duplicated")]
                    )
                result[key] = _construct_node(loader, value_node, key_path, active_nodes, seen_nodes)
            return result
        if isinstance(node, SequenceNode):
            return [
                _construct_node(loader, child, path + (index,), active_nodes, seen_nodes)
                for index, child in enumerate(node.value)
            ]
        if isinstance(node, ScalarNode):
            return loader.construct_object(node, deep=True)
        raise AutonomousValidationError(
            [ValidationIssue(path, "unsupported_yaml_node", f"unsupported YAML node {type(node).__name__}")]
        )
    finally:
        active_nodes.remove(node_id)
