# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build arm-local and full-state cuRobo configurations from a named URDF state."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import tempfile
import xml.etree.ElementTree as ET
import yaml
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .collision_spheres import CollisionSphere, CollisionSphereSet

ATTACHED_OBJECT_LINK_NAME = "attached_object"


@dataclass(frozen=True)
class CuroboArmChain:
    """URDF-derived kinematic ownership for one independently planned tool.

    Args:
        root_link: Planning reference link.
        tool_link: Controlled end-effector link.
        root_to_tool_link_names: Ordered links along the serial chain.
        auxiliary_joint_names: Additional independent joints explicitly assigned to the group.
        active_joint_names: Serial and auxiliary joints in planner order.
        moving_link_names: Descendants of active joints, including other branches when shared.
        collision_link_names: Geometry-bearing links owned by this tool's chain and contact subtree.
        contact_link_names: Tool link and descendants used to resolve hand contact ownership.
    """

    root_link: str
    tool_link: str
    root_to_tool_link_names: tuple[str, ...]
    auxiliary_joint_names: tuple[str, ...]
    active_joint_names: tuple[str, ...]
    moving_link_names: tuple[str, ...]
    collision_link_names: tuple[str, ...]
    contact_link_names: tuple[str, ...]


@dataclass(frozen=True)
class _Joint:
    """Named URDF connection with its type, endpoints and optional mimic leader."""

    name: str
    joint_type: str
    parent_link: str
    child_link: str
    mimic_joint: str | None


@dataclass(frozen=True)
class _UrdfModel:
    """Parsed single-root topology and the names of links containing geometry."""

    links: tuple[str, ...]
    geometry_link_names: tuple[str, ...]
    joints: Mapping[str, _Joint]
    root_link: str


def curobo_robot_model_prepare(
    urdf_path: str | Path,
    *,
    auxiliary_joint_names: Sequence[str] = (),
) -> Path:
    """Materialize the planning view of a controller-owned URDF.

    Some task-space controllers reduce their own URDF by marking joints handled by a separate
    positional action channel as fixed. A planning group may explicitly restore those joints as
    movable. The derived file stays beside the source URDF so relative mesh paths remain valid.

    Args:
        urdf_path: Controller-owned local URDF.
        auxiliary_joint_names: Independently commanded joints to restore when fixed.

    Returns:
        Source path when no restoration is needed, otherwise an atomically cached derived URDF.
    """

    source_path = Path(urdf_path).expanduser().resolve()
    assert source_path.is_file(), f"URDF must be a local file: {source_path}"
    requested = tuple(auxiliary_joint_names)
    assert all(isinstance(name, str) and name for name in requested)
    assert len(requested) == len(set(requested)), "auxiliary_joint_names must be unique"

    root = ET.fromstring(source_path.read_bytes())
    joint_elements = {element.get("name", ""): element for element in root.findall("joint")}
    unknown = sorted(set(requested) - set(joint_elements))
    assert not unknown, f"unknown auxiliary joints: {unknown}"
    restored: list[str] = []
    for name in requested:
        element = joint_elements[name]
        assert element.find("mimic") is None, f"auxiliary joint {name!r} is mimic-controlled"
        joint_type = element.get("type", "")
        if joint_type == "fixed":
            assert (
                element.find("axis") is not None and element.find("limit") is not None
            ), f"fixed auxiliary joint {name!r} lacks the axis/limit metadata needed for restoration"
            element.set("type", "revolute")
            restored.append(name)
        else:
            assert joint_type in {
                "continuous",
                "prismatic",
                "revolute",
            }, f"unsupported auxiliary joint type {joint_type!r} for {name!r}"
    if not restored:
        return source_path

    digest = hashlib.sha256()
    digest.update(source_path.read_bytes())
    digest.update("\0".join(sorted(requested)).encode())
    output_path = source_path.with_name(f"{source_path.stem}.planning-{digest.hexdigest()[:16]}.urdf")
    if output_path.is_file():
        return output_path
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            ET.ElementTree(root).write(stream, encoding="utf-8", xml_declaration=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return output_path


def curobo_arm_chain(
    urdf_path: str | Path,
    tool_link: str,
    *,
    root_link: str | None = None,
    inactive_joint_names: Sequence[str] = (),
    auxiliary_joint_names: Sequence[str] = (),
) -> CuroboArmChain:
    """Derive active joints and moving links for one root-to-tool chain.

    Joints on the chain are active unless fixed, mimic-controlled, or explicitly inactive.
    Independently commanded joints below that chain, such as dexterous-hand joints, may be added
    explicitly. Moving links are the child and descendants of every active joint. Collision
    ownership is narrower: geometry on the root-to-tool branch plus the selected tool's contact
    descendants. This prevents a shared humanoid joint from making one group own its sibling arm.

    Args:
        urdf_path: Local URDF describing the complete robot.
        tool_link: End-effector link whose chain is extracted.
        root_link: Planning reference link; ``None`` uses the URDF root.
        inactive_joint_names: Joints to exclude from the active serial chain.
        auxiliary_joint_names: Independent joints outside the serial chain to activate explicitly.

    Returns:
        Named joint, motion, collision and contact ownership for the planning group.
    """

    model = _urdf_model_load(Path(urdf_path))
    root_name = model.root_link if root_link is None else root_link
    assert root_name in model.links, f"unknown root link {root_name!r}"
    assert tool_link in model.links, f"unknown tool link {tool_link!r}"
    inactive = set(inactive_joint_names)
    unknown_inactive = sorted(inactive - set(model.joints))
    assert not unknown_inactive, f"unknown inactive joints: {unknown_inactive}"
    auxiliary = tuple(auxiliary_joint_names)
    assert all(
        isinstance(name, str) and name for name in auxiliary
    ), "auxiliary_joint_names must contain non-empty strings"
    assert len(auxiliary) == len(set(auxiliary)), "auxiliary_joint_names must be unique"
    unknown_auxiliary = sorted(set(auxiliary) - set(model.joints))
    assert not unknown_auxiliary, f"unknown auxiliary joints: {unknown_auxiliary}"

    joint_by_child = {joint.child_link: joint for joint in model.joints.values()}
    reverse_chain: list[_Joint] = []
    cursor = tool_link
    while cursor != root_name:
        assert cursor in joint_by_child, f"tool link {tool_link!r} is not below root {root_name!r}"
        joint = joint_by_child[cursor]
        reverse_chain.append(joint)
        cursor = joint.parent_link
    chain = tuple(reversed(reverse_chain))
    serial_active_joints = tuple(
        joint.name
        for joint in chain
        if joint.joint_type != "fixed" and joint.mimic_joint is None and joint.name not in inactive
    )
    assert serial_active_joints, f"root-to-tool chain for {tool_link!r} has no active joints"

    redundant_auxiliary = sorted(set(auxiliary) & {joint.name for joint in chain})
    assert not redundant_auxiliary, f"auxiliary joints already belong to the root-to-tool chain: {redundant_auxiliary}"
    inactive_auxiliary = sorted(set(auxiliary) & inactive)
    assert not inactive_auxiliary, f"auxiliary joints are explicitly inactive: {inactive_auxiliary}"
    unsupported_auxiliary = sorted(
        name
        for name in auxiliary
        if model.joints[name].joint_type == "fixed" or model.joints[name].mimic_joint is not None
    )
    assert not unsupported_auxiliary, f"auxiliary joints must be independent and non-fixed: {unsupported_auxiliary}"

    children = _children_by_link(model)
    serial_moving: set[str] = set()
    for joint_name in serial_active_joints:
        queue = deque([model.joints[joint_name].child_link])
        while queue:
            link_name = queue.popleft()
            if link_name in serial_moving:
                continue
            serial_moving.add(link_name)
            queue.extend(children[link_name])
    outside_branch = sorted(name for name in auxiliary if model.joints[name].child_link not in serial_moving)
    assert not outside_branch, f"auxiliary joints do not belong to the selected tool branch: {outside_branch}"

    active_joints = (*serial_active_joints, *auxiliary)
    moving: set[str] = set()
    for joint_name in active_joints:
        queue = deque([model.joints[joint_name].child_link])
        while queue:
            link_name = queue.popleft()
            if link_name in moving:
                continue
            moving.add(link_name)
            queue.extend(children[link_name])
    moving_link_names = tuple(link_name for link_name in model.links if link_name in moving)
    collision_links = set(model.geometry_link_names)
    contact_joints = set(auxiliary)
    while (
        followers := {joint.name for joint in model.joints.values() if joint.mimic_joint in contact_joints}
        - contact_joints
    ):
        contact_joints.update(followers)
    contact_links: set[str] = set()
    queue = deque([tool_link, *(model.joints[name].child_link for name in contact_joints)])
    while queue:
        link_name = queue.popleft()
        if link_name in contact_links:
            continue
        contact_links.add(link_name)
        queue.extend(children[link_name])
    owned_collision_links = ({root_name, *(joint.child_link for joint in chain)} | contact_links) & collision_links
    return CuroboArmChain(
        root_link=root_name,
        tool_link=tool_link,
        root_to_tool_link_names=(root_name, *(joint.child_link for joint in chain)),
        auxiliary_joint_names=auxiliary,
        active_joint_names=active_joints,
        moving_link_names=moving_link_names,
        collision_link_names=tuple(link_name for link_name in model.links if link_name in owned_collision_links),
        contact_link_names=tuple(link_name for link_name in moving_link_names if link_name in contact_links),
    )


def curobo_robot_config_create(
    urdf_path: str | Path,
    tool_link: str,
    start_joint_positions: Mapping[str, float],
    collision_spheres: CollisionSphereSet | Mapping[str, Sequence[CollisionSphere | Mapping[str, object]]],
    *,
    root_link: str | None = None,
    inactive_joint_names: Sequence[str] = (),
    auxiliary_joint_names: Sequence[str] = (),
    lock_auxiliary_joints: bool = False,
    collision_link_names: Sequence[str] | None = None,
    attachment_sphere_count: int = 50,
    self_collision_ignore_depth: int = 2,
    self_collision_ignore_pairs: Sequence[Sequence[str]] = (),
    max_acceleration: float = 15.0,
    max_jerk: float = 500.0,
) -> dict[str, object]:
    """Create an ordinary arm-local cuRobo ``robot_cfg`` mapping.

    Inactive joints inside the selected root-to-tool chain and contact subtree are locked to the
    supplied named start state. Unowned branches are omitted entirely. Only requested collision
    spheres owned by this tool are emitted. Carried-object capacity lives on an
    identity-fixed virtual child of the tool so cuRobo never replaces the tool's fitted geometry.

    Args:
        urdf_path: Local URDF describing the complete robot.
        tool_link: End-effector link whose planning configuration is built.
        start_joint_positions: Named initial state covering independent owned joints [rad, m].
        collision_spheres: Shared link-frame geometry to project onto this group [m].
        root_link: Planning reference link; ``None`` uses the URDF root.
        inactive_joint_names: Serial-chain joints locked to the supplied initial state.
        auxiliary_joint_names: Independent joints outside the serial chain to model explicitly.
        lock_auxiliary_joints: Hold auxiliary joints at their named start positions during planning.
        collision_link_names: Optional subset of this tool's geometry-bearing links.
        attachment_sphere_count: Reserved carried-object sphere slots; zero disables attachment slots.
        self_collision_ignore_depth: Active-joint topology distance excluded from self-collision checks.
        self_collision_ignore_pairs: Additional explicit link-pair exclusions.
        max_acceleration: Planner acceleration limit [rad/s^2, m/s^2].
        max_jerk: Planner jerk limit [rad/s^3, m/s^3].

    Returns:
        A mapping accepted by cuRobo's robot configuration loader.
    """

    source_path = Path(urdf_path).expanduser().resolve()
    model = _urdf_model_load(source_path)
    chain = curobo_arm_chain(
        source_path,
        tool_link,
        root_link=root_link,
        inactive_joint_names=inactive_joint_names,
        auxiliary_joint_names=auxiliary_joint_names,
    )
    owned_links = set(chain.root_to_tool_link_names) | set(chain.contact_link_names)
    independent = tuple(
        joint.name
        for joint in model.joints.values()
        if joint.joint_type != "fixed" and joint.mimic_joint is None and joint.child_link in owned_links
    )
    source_spheres = (
        collision_spheres.spheres_by_link if isinstance(collision_spheres, CollisionSphereSet) else collision_spheres
    )
    requested_collision_links = (
        chain.collision_link_names if collision_link_names is None else tuple(collision_link_names)
    )
    unknown_collision_links = set(requested_collision_links) - set(chain.collision_link_names)
    assert (
        not unknown_collision_links
    ), f"collision links are not owned by tool {tool_link!r}: {sorted(unknown_collision_links)}"
    selected_collision_links = tuple(name for name in requested_collision_links if name in source_spheres)
    assert selected_collision_links, f"no fitted collision links belong to tool {tool_link!r}"
    if tool_link not in selected_collision_links:
        selected_collision_links = (*selected_collision_links, tool_link)
    return _robot_config_create(
        source_path,
        model,
        root_link=chain.root_link,
        link_names=(tool_link,),
        independent_joint_names=independent,
        active_joint_names=tuple(
            name for name in chain.active_joint_names if not lock_auxiliary_joints or name not in auxiliary_joint_names
        ),
        collision_link_names=selected_collision_links,
        start_joint_positions=start_joint_positions,
        source_spheres=source_spheres,
        attachment_parent_link=tool_link if attachment_sphere_count else None,
        attachment_sphere_count=attachment_sphere_count,
        self_collision_ignore_depth=self_collision_ignore_depth,
        self_collision_ignore_pairs=self_collision_ignore_pairs,
        max_acceleration=max_acceleration,
        max_jerk=max_jerk,
    )


def curobo_full_robot_config_create(
    urdf_path: str | Path,
    tool_links: Sequence[str],
    active_joint_names: Sequence[str],
    start_joint_positions: Mapping[str, float],
    collision_spheres: CollisionSphereSet | Mapping[str, Sequence[CollisionSphere | Mapping[str, object]]],
    *,
    root_link: str | None = None,
    self_collision_ignore_depth: int = 2,
    self_collision_ignore_pairs: Sequence[Sequence[str]] = (),
    max_acceleration: float = 15.0,
    max_jerk: float = 500.0,
) -> dict[str, object]:
    """Create a cuRobo configuration for composed robot-state checks.

    Every independent joint not owned by a planning group is locked to the exact named start
    state. Supplied fitted links form the collision model, while motion planning and IK stay on the
    ordinary arm-local configurations.

    Args:
        urdf_path: Controller-owned local URDF.
        tool_links: Tool links exposed by the full-state FK model.
        active_joint_names: Union of dynamically composed planning-group joints.
        start_joint_positions: Complete named live articulation state [rad, m].
        collision_spheres: Shared collision spheres for the links to check.
        root_link: Optional planning root; the URDF root is used by default.
        self_collision_ignore_depth: Active-joint topology depth ignored for self collision.
        self_collision_ignore_pairs: Additional reviewed link pairs to ignore.
        max_acceleration: cuRobo configuration acceleration limit [rad/s^2, m/s^2].
        max_jerk: cuRobo configuration jerk limit [rad/s^3, m/s^3].

    Returns:
        A mapping accepted by cuRobo's robot configuration loader.
    """

    source_path = Path(urdf_path).expanduser().resolve()
    model = _urdf_model_load(source_path)
    root_name = model.root_link if root_link is None else root_link
    assert root_name in model.links, f"unknown root link {root_name!r}"
    tools = tuple(tool_links)
    assert tools and all(name in model.links for name in tools), "tool_links must name URDF links"
    assert len(tools) == len(set(tools)), "tool_links must be unique"

    requested_active = set(active_joint_names)
    assert requested_active, "active_joint_names must not be empty"
    unknown_active = sorted(requested_active - set(model.joints))
    assert not unknown_active, f"unknown active joints: {unknown_active}"
    unsupported_active = sorted(
        name
        for name in requested_active
        if model.joints[name].joint_type == "fixed" or model.joints[name].mimic_joint is not None
    )
    assert not unsupported_active, f"active joints must be independent and non-fixed: {unsupported_active}"
    active = tuple(joint.name for joint in model.joints.values() if joint.name in requested_active)
    source_spheres = (
        collision_spheres.spheres_by_link if isinstance(collision_spheres, CollisionSphereSet) else collision_spheres
    )
    collision_link_names = tuple(name for name in model.geometry_link_names if name in source_spheres)
    assert collision_link_names, "full-state collision model must contain at least one robot link"
    scoped_links = set(model.links)
    if set(collision_link_names) != set(model.geometry_link_names):
        scoped_links = {root_name}
        joint_by_child = {joint.child_link: joint for joint in model.joints.values()}
        for target_link in (*tools, *collision_link_names):
            cursor = target_link
            while cursor != root_name:
                assert cursor in joint_by_child, f"link {target_link!r} is not below root {root_name!r}"
                scoped_links.add(cursor)
                cursor = joint_by_child[cursor].parent_link
    independent = tuple(
        joint.name
        for joint in model.joints.values()
        if joint.joint_type != "fixed" and joint.mimic_joint is None and joint.child_link in scoped_links
    )
    assert requested_active <= set(independent), "active joints must belong to the scoped collision/tool model"
    return _robot_config_create(
        source_path,
        model,
        root_link=root_name,
        link_names=tools,
        independent_joint_names=independent,
        active_joint_names=active,
        collision_link_names=collision_link_names,
        start_joint_positions=start_joint_positions,
        source_spheres=source_spheres,
        attachment_parent_link=None,
        attachment_sphere_count=0,
        self_collision_ignore_depth=self_collision_ignore_depth,
        self_collision_ignore_pairs=self_collision_ignore_pairs,
        max_acceleration=max_acceleration,
        max_jerk=max_jerk,
    )


def _robot_config_create(
    source_path: Path,
    model: _UrdfModel,
    *,
    root_link: str,
    link_names: tuple[str, ...],
    independent_joint_names: tuple[str, ...],
    active_joint_names: tuple[str, ...],
    collision_link_names: tuple[str, ...],
    start_joint_positions: Mapping[str, float],
    source_spheres: Mapping[str, Sequence[CollisionSphere | Mapping[str, object]]],
    attachment_parent_link: str | None,
    attachment_sphere_count: int,
    self_collision_ignore_depth: int,
    self_collision_ignore_pairs: Sequence[Sequence[str]],
    max_acceleration: float,
    max_jerk: float,
) -> dict[str, object]:
    """Assemble the shared cuRobo kinematics mapping."""

    assert attachment_sphere_count >= 0
    assert self_collision_ignore_depth >= 0
    assert math.isfinite(max_acceleration) and max_acceleration > 0.0
    assert math.isfinite(max_jerk) and max_jerk > 0.0
    assert set(active_joint_names) <= set(independent_joint_names)
    missing_positions = sorted(set(independent_joint_names) - set(start_joint_positions))
    assert not missing_positions, f"start state is missing independent joints: {missing_positions}"
    positions = {name: float(start_joint_positions[name]) for name in independent_joint_names}
    assert all(math.isfinite(value) for value in positions.values()), "start joint positions must be finite"
    active = set(active_joint_names)
    lock_joints = {name: positions[name] for name in independent_joint_names if name not in active}
    projected_spheres = {
        link_name: [_sphere_dict(sphere) for sphere in source_spheres.get(link_name, ())]
        for link_name in collision_link_names
    }
    self_collision_ignore = _self_collision_ignore(
        model,
        active_joint_names,
        collision_link_names,
        self_collision_ignore_depth,
    )
    _self_collision_ignore_pairs_add(
        self_collision_ignore,
        collision_link_names,
        self_collision_ignore_pairs,
    )
    extra_links: dict[str, object] = {}
    if attachment_parent_link is not None:
        attachment_joint_name = f"{ATTACHED_OBJECT_LINK_NAME}_joint"
        assert (
            ATTACHED_OBJECT_LINK_NAME not in model.links
        ), f"reserved attachment link {ATTACHED_OBJECT_LINK_NAME!r} already exists in the URDF"
        assert (
            attachment_joint_name not in model.joints
        ), f"reserved attachment joint {attachment_joint_name!r} already exists in the URDF"
        attachment_ignores = [attachment_parent_link, *self_collision_ignore[attachment_parent_link]]
        self_collision_ignore[ATTACHED_OBJECT_LINK_NAME] = attachment_ignores
        for link_name in attachment_ignores:
            self_collision_ignore[link_name].append(ATTACHED_OBJECT_LINK_NAME)
        collision_link_names = (*collision_link_names, ATTACHED_OBJECT_LINK_NAME)
        projected_spheres[ATTACHED_OBJECT_LINK_NAME] = [
            {"center": [0.0, 0.0, 0.0], "radius": -10.0} for _ in range(attachment_sphere_count)
        ]
        extra_links[ATTACHED_OBJECT_LINK_NAME] = {
            "parent_link_name": attachment_parent_link,
            "link_name": ATTACHED_OBJECT_LINK_NAME,
            "fixed_transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "joint_type": "FIXED",
            "joint_name": attachment_joint_name,
        }
    kinematics: dict[str, object] = {
        "use_usd_kinematics": False,
        "urdf_path": str(source_path),
        "asset_root_path": str(source_path.parent),
        "base_link": root_link,
        "ee_link": link_names[0],
        "link_names": list(link_names),
        "collision_link_names": list(collision_link_names),
        "collision_spheres": projected_spheres,
        "collision_sphere_buffer": 0.0,
        "self_collision_ignore": self_collision_ignore,
        "self_collision_buffer": {link_name: 0.0 for link_name in collision_link_names},
        "lock_joints": lock_joints,
        "extra_links": extra_links,
        "extra_collision_spheres": {},
        "cspace": {
            "joint_names": list(active_joint_names),
            "retract_config": [positions[name] for name in active_joint_names],
            "null_space_weight": [1.0] * len(active_joint_names),
            "cspace_distance_weight": [1.0] * len(active_joint_names),
            "max_acceleration": max_acceleration,
            "max_jerk": max_jerk,
        },
    }
    return {"robot_cfg": {"kinematics": kinematics}}


def curobo_robot_config_write(config: Mapping[str, object], path: str | Path) -> Path:
    """Atomically write a cuRobo robot configuration YAML file.

    Args:
        config: Robot configuration mapping containing ``robot_cfg``.
        path: Destination YAML file; parent directories are created if needed.

    Returns:
        Resolved destination path.
    """

    assert "robot_cfg" in config, "config must contain robot_cfg"
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(copy.deepcopy(dict(config)), stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return output_path


def _urdf_model_load(path: Path) -> _UrdfModel:
    """Read named links, geometry presence and joint topology from a single-root URDF."""

    source_path = path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"URDF must be a local file: {source_path}")
    root = ET.fromstring(source_path.read_bytes())
    link_elements = root.findall("link")
    links = tuple(element.get("name", "") for element in link_elements)
    assert links and all(links) and len(set(links)) == len(links), "URDF links must be named and unique"
    geometry_link_names = tuple(
        element.get("name", "")
        for element in link_elements
        if element.find("visual") is not None or element.find("collision") is not None
    )
    joints: dict[str, _Joint] = {}
    for element in root.findall("joint"):
        parent, child = element.find("parent"), element.find("child")
        name = element.get("name", "")
        assert name and name not in joints and parent is not None and child is not None
        mimic = element.find("mimic")
        joints[name] = _Joint(
            name=name,
            joint_type=element.get("type", ""),
            parent_link=parent.get("link", ""),
            child_link=child.get("link", ""),
            mimic_joint=None if mimic is None else mimic.get("joint"),
        )
    child_links = {joint.child_link for joint in joints.values()}
    roots = tuple(link_name for link_name in links if link_name not in child_links)
    assert len(roots) == 1, f"URDF must have one root link, found {roots}"
    return _UrdfModel(
        links=links,
        geometry_link_names=geometry_link_names,
        joints=joints,
        root_link=roots[0],
    )


def _children_by_link(model: _UrdfModel) -> dict[str, list[str]]:
    """Index child links by their parent link in URDF joint order."""

    children: dict[str, list[str]] = defaultdict(list)
    for joint in model.joints.values():
        children[joint.parent_link].append(joint.child_link)
    return children


def _self_collision_ignore(
    model: _UrdfModel,
    active_joint_names: tuple[str, ...],
    collision_link_names: tuple[str, ...],
    depth: int,
) -> dict[str, list[str]]:
    """Derive symmetric collision exclusions from the active-joint topology.

    Args:
        model: Full URDF link and joint topology.
        active_joint_names: Joints that remain movable in this planning configuration.
        collision_link_names: Links whose sphere pairs participate in collision checking.
        depth: Maximum active-joint graph distance to exclude after merging inactive joints.

    Returns:
        Ignored partners for each collision link. Links in the same rigid cluster are always
        excluded; these are topology-based exclusions, not sampled collision observations.
    """

    parent = {link_name: link_name for link_name in model.links}

    def find(link_name: str) -> str:
        """Return the rigid-cluster representative, compressing traversed parent links."""

        while parent[link_name] != link_name:
            parent[link_name] = parent[parent[link_name]]
            link_name = parent[link_name]
        return link_name

    def union(first: str, second: str) -> None:
        """Merge two link clusters connected by an inactive joint."""

        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    active = set(active_joint_names)
    for joint in model.joints.values():
        if joint.name not in active:
            union(joint.parent_link, joint.child_link)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for joint_name in active_joint_names:
        joint = model.joints[joint_name]
        first, second = find(joint.parent_link), find(joint.child_link)
        adjacency[first].add(second)
        adjacency[second].add(first)

    result: dict[str, list[str]] = {link_name: [] for link_name in collision_link_names}
    for index, first_link in enumerate(collision_link_names):
        distances = {find(first_link): 0}
        queue = deque(distances)
        while queue:
            cluster = queue.popleft()
            if distances[cluster] == depth:
                continue
            for neighbor in adjacency[cluster]:
                if neighbor not in distances:
                    distances[neighbor] = distances[cluster] + 1
                    queue.append(neighbor)
        for second_link in collision_link_names[index + 1 :]:
            if distances.get(find(second_link), depth + 1) <= depth:
                result[first_link].append(second_link)
                result[second_link].append(first_link)
    return result


def _self_collision_ignore_pairs_add(
    ignore_by_link: dict[str, list[str]],
    collision_link_names: tuple[str, ...],
    pairs: Sequence[Sequence[str]],
) -> None:
    """Validate explicit link pairs and add symmetric, canonically ordered exclusions."""

    link_order = {name: index for index, name in enumerate(collision_link_names)}
    link_names = set(link_order)
    normalized: set[tuple[str, str]] = set()
    for pair in pairs:
        assert not isinstance(pair, str) and len(pair) == 2, "self-collision ignore pairs must contain two links"
        first, second = pair
        assert isinstance(first, str) and isinstance(second, str) and first and second
        assert first != second, f"self-collision ignore pair repeats link {first!r}"
        unknown = sorted({first, second} - link_names)
        assert not unknown, f"self-collision ignore pair contains links outside the planning group: {unknown}"
        if link_order[first] > link_order[second]:
            first, second = second, first
        assert (first, second) not in normalized, f"duplicate self-collision ignore pair: {(first, second)}"
        normalized.add((first, second))

    for first, second in sorted(normalized, key=lambda pair: (link_order[pair[0]], link_order[pair[1]])):
        if second not in ignore_by_link[first]:
            ignore_by_link[first].append(second)
        if first not in ignore_by_link[second]:
            ignore_by_link[second].append(first)
    for ignored_links in ignore_by_link.values():
        ignored_links.sort(key=link_order.__getitem__)


def _sphere_dict(sphere: CollisionSphere | Mapping[str, object]) -> dict[str, object]:
    """Validate and serialize one sphere as a cuRobo-compatible center/radius mapping."""

    if isinstance(sphere, CollisionSphere):
        return sphere.to_dict()
    center = tuple(float(value) for value in sphere["center"])
    radius = float(sphere["radius"])
    return CollisionSphere(center=center, radius=radius).to_dict()
