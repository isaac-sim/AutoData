# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "docker" / "run_autonomous_task.sh"
CONTAINER_ENTRYPOINT = REPO_ROOT / "docker" / "autonomous_entrypoint.sh"
IMAGE_ID = f"sha256:{'a' * 64}"
SCHEDULESTREAM_COMMIT = "b" * 40
BASE_IMAGE_ID = f"sha256:{'c' * 64}"
V1_IMAGE_CONTRACT = f"custream|v1|{SCHEDULESTREAM_COMMIT}|{BASE_IMAGE_ID}"


def _write_fake_docker(root: Path) -> tuple[Path, Path, Path]:
    bin_dir = root / "bin"
    bin_dir.mkdir()
    run_args_path = root / "docker-run-args.bin"
    inspect_args_path = root / "docker-inspect-args.bin"
    docker_path = bin_dir / "docker"
    docker_path.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        ': "${FAKE_DOCKER_RUN_ARGS:?}" "${FAKE_DOCKER_INSPECT_ARGS:?}"\n'
        'if [[ "$1" == "image" && "$2" == "inspect" ]]; then\n'
        '    printf \'%s\\0\' "$@" >>"${FAKE_DOCKER_INSPECT_ARGS}"\n'
        "    printf '\\0' >>\"${FAKE_DOCKER_INSPECT_ARGS}\"\n"
        '    if [[ "$4" == "{{.Id}}" ]]; then\n'
        "        printf '%s\\n' \"${FAKE_DOCKER_IMAGE_ID:?}\"\n"
        "    else\n"
        "        printf '%s\\n' \"${FAKE_DOCKER_IMAGE_CONTRACT:?}\"\n"
        "    fi\n"
        "    exit 0\n"
        "fi\n"
        'if [[ "$1" == "run" ]]; then\n'
        '    printf \'%s\\0\' "$@" >"${FAKE_DOCKER_RUN_ARGS}"\n'
        '    exit "${FAKE_DOCKER_EXIT:-0}"\n'
        "fi\n"
        "printf 'unexpected docker invocation: %s\\n' \"$*\" >&2\n"
        "exit 97\n",
        encoding="utf-8",
    )
    docker_path.chmod(0o755)
    return bin_dir, run_args_path, inspect_args_path


def _write_fake_xauth(bin_dir: Path) -> None:
    xauth_path = bin_dir / "xauth"
    xauth_path.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'case "$3" in\n'
        "    nlist) printf '0000ffff0123456789abcdef\\n' ;;\n"
        '    nmerge) cat >"$2" ;;\n'
        "    *) exit 2 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    xauth_path.chmod(0o755)


def _task_yaml(root: Path) -> Path:
    task_path = root / "task.yaml"
    task_path.write_text("schema_version: 1\n", encoding="utf-8")
    return task_path


def _run_launcher(
    root: Path,
    *arguments: str,
    launcher: Path = LAUNCHER,
    accept_eula: str | None = "Y",
    docker_exit: int = 0,
    image_contract: str = V1_IMAGE_CONTRACT,
    fake_xauth: bool = False,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str], list[list[str]]]:
    bin_dir, run_args_path, inspect_args_path = _write_fake_docker(root)
    if fake_xauth:
        _write_fake_xauth(bin_dir)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_DOCKER_RUN_ARGS"] = str(run_args_path)
    env["FAKE_DOCKER_INSPECT_ARGS"] = str(inspect_args_path)
    env["FAKE_DOCKER_EXIT"] = str(docker_exit)
    env["FAKE_DOCKER_IMAGE_ID"] = IMAGE_ID
    env["FAKE_DOCKER_IMAGE_CONTRACT"] = image_contract
    if accept_eula is None:
        env.pop("ACCEPT_EULA", None)
    else:
        env["ACCEPT_EULA"] = accept_eula
    if extra_env is not None:
        env.update(extra_env)

    result = subprocess.run(
        [str(launcher), *arguments],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    docker_run_args = []
    if run_args_path.exists():
        encoded_args = run_args_path.read_bytes().split(b"\0")
        docker_run_args = [value.decode() for value in encoded_args if value]

    docker_inspect_args = []
    if inspect_args_path.exists():
        for encoded_call in inspect_args_path.read_bytes().split(b"\0\0"):
            if encoded_call:
                docker_inspect_args.append([value.decode() for value in encoded_call.split(b"\0") if value])
    return result, docker_run_args, docker_inspect_args


def _option_values(arguments: list[str], option: str) -> list[str]:
    return [arguments[index + 1] for index, value in enumerate(arguments[:-1]) if value == option]


def test_headless_launch_has_scoped_mounts_and_v1_caches(tmp_path: Path) -> None:
    task_path = _task_yaml(tmp_path)
    run_dir = tmp_path / "run"

    result, docker_args, inspect_calls = _run_launcher(tmp_path, "--run-dir", str(run_dir), str(task_path))

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"Run directory: {run_dir.resolve()}\n"
    assert [call[-1] for call in inspect_calls] == ["isaac_autodata:schedulestream-v1", IMAGE_ID]
    assert docker_args[0] == "run"
    assert _option_values(docker_args, "--security-opt") == ["no-new-privileges:true"]
    assert _option_values(docker_args, "--user") == ["0:0"]
    assert _option_values(docker_args, "--entrypoint") == ["/workspaces/isaac_autodata/docker/autonomous_entrypoint.sh"]

    env_values = _option_values(docker_args, "--env")
    assert "ACCEPT_EULA" in env_values
    assert "ACCEPT_EULA=Y" not in env_values
    assert f"DOCKER_RUN_USER_ID={os.getuid()}" in env_values
    assert f"DOCKER_RUN_GROUP_ID={os.getgid()}" in env_values

    mounts = _option_values(docker_args, "--mount")
    assert f"type=bind,src={REPO_ROOT},dst=/workspaces/isaac_autodata,readonly" in mounts
    assert f"type=bind,src={run_dir.resolve()},dst=/autonomous-run" in mounts
    task_copy = run_dir / "task.yaml"
    assert f"type=bind,src={task_copy.resolve()},dst=/autonomous-run/task.yaml,readonly" in mounts
    assert task_copy.read_bytes() == task_path.read_bytes()
    assert task_copy.stat().st_uid == os.getuid()
    assert task_copy.stat().st_mode & 0o777 == 0o400

    cache_namespace = f"isaac-autodata-isaac-sim-5-1-v1-{IMAGE_ID[7:19]}-u{os.getuid()}"
    assert f"type=volume,src={cache_namespace}-kit,dst=/isaac-sim/kit/cache" in mounts
    assert f"type=volume,src={cache_namespace}-ov,dst=/autodata-home/.cache/ov" in mounts
    assert f"type=volume,src={cache_namespace}-warp,dst=/autodata-home/.cache/warp" in mounts
    assert f"type=volume,src={cache_namespace}-gl,dst=/autodata-home/.cache/nvidia/GLCache" in mounts
    assert f"type=volume,src={cache_namespace}-compute,dst=/autodata-home/.nv/ComputeCache" in mounts
    assert all("umi" not in argument.lower() for argument in docker_args)

    assert docker_args[-4:] == [
        IMAGE_ID,
        "/isaac-sim/python.sh",
        "/workspaces/isaac_autodata/isaac_autodata_examples/generate_task_dataset.py",
        "/autonomous-run/task.yaml",
    ]
    assert "--gui" not in docker_args
    assert not any("/tmp/.X11-unix" in mount for mount in mounts)


def test_gui_maps_x11_and_forwards_gui_flag(tmp_path: Path) -> None:
    task_path = _task_yaml(tmp_path)
    run_dir = tmp_path / "run"
    host_xauthority = tmp_path / "host.Xauthority"
    host_xauthority.write_text("host cookie database\n", encoding="utf-8")
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()

    x11_socket = socket.socket(socket.AF_UNIX)
    x11_socket_path = None
    try:
        for display_number in range(200, 1000):
            candidate = Path(f"/tmp/.X11-unix/X{display_number}")
            try:
                x11_socket.bind(str(candidate))
            except OSError:
                continue
            x11_socket_path = candidate
            break
        assert x11_socket_path is not None
        display = f":{x11_socket_path.name.removeprefix('X')}"

        result, docker_args, _ = _run_launcher(
            tmp_path,
            "--gui",
            "--run-dir",
            str(run_dir),
            str(task_path),
            fake_xauth=True,
            extra_env={
                "DISPLAY": display,
                "XAUTHORITY": str(host_xauthority),
                "XDG_RUNTIME_DIR": str(runtime_dir),
            },
        )
    finally:
        x11_socket.close()
        if x11_socket_path is not None:
            x11_socket_path.unlink(missing_ok=True)

    assert result.returncode == 0, result.stderr
    env_values = _option_values(docker_args, "--env")
    assert f"DISPLAY={display}" in env_values
    assert "XAUTHORITY=/autodata-xauthority" in env_values
    assert "QT_X11_NO_MITSHM=1" in env_values
    mounts = _option_values(docker_args, "--mount")
    assert "type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly" in mounts
    xauthority_mount = next(mount for mount in mounts if mount.endswith("dst=/autodata-xauthority,readonly"))
    minimized_xauthority = Path(xauthority_mount.removeprefix("type=bind,src=").split(",dst=", maxsplit=1)[0])
    assert not minimized_xauthority.exists()
    assert list(runtime_dir.iterdir()) == []
    assert docker_args[-1] == "--gui"


def test_eula_must_be_explicit_before_run_directory_creation(tmp_path: Path) -> None:
    task_path = _task_yaml(tmp_path)
    run_dir = tmp_path / "run"

    result, docker_args, inspect_calls = _run_launcher(
        tmp_path,
        "--run-dir",
        str(run_dir),
        str(task_path),
        accept_eula=None,
    )

    assert result.returncode == 2
    assert "ACCEPT_EULA=Y" in result.stderr
    assert docker_args == []
    assert inspect_calls == []
    assert not run_dir.exists()


def test_docker_exit_code_is_returned_after_printing_run_directory(tmp_path: Path) -> None:
    task_path = _task_yaml(tmp_path)
    run_dir = tmp_path / "run"

    result, docker_args, _ = _run_launcher(
        tmp_path,
        "--run-dir",
        str(run_dir),
        str(task_path),
        docker_exit=37,
    )

    assert result.returncode == 37
    assert result.stdout == f"Run directory: {run_dir.resolve()}\n"
    assert docker_args[0] == "run"


def test_invalid_image_labels_are_rejected_before_docker_run(tmp_path: Path) -> None:
    task_path = _task_yaml(tmp_path)
    run_dir = tmp_path / "run"

    invalid_contract = f"custream2|v1|{SCHEDULESTREAM_COMMIT}|{BASE_IMAGE_ID}"
    result, docker_args, inspect_calls = _run_launcher(
        tmp_path,
        "--run-dir",
        str(run_dir),
        str(task_path),
        image_contract=invalid_contract,
    )

    assert result.returncode == 2
    assert "image is not a supported ScheduleStream/cuRobo runtime" in result.stderr
    assert docker_args == []
    assert len(inspect_calls) == 2
    assert not run_dir.exists()


def test_default_run_directory_is_unique_and_below_dataset_root(tmp_path: Path) -> None:
    fake_repo = tmp_path / "repo"
    docker_dir = fake_repo / "docker"
    docker_dir.mkdir(parents=True)
    launcher = docker_dir / LAUNCHER.name
    entrypoint = docker_dir / CONTAINER_ENTRYPOINT.name
    shutil.copy2(LAUNCHER, launcher)
    shutil.copy2(CONTAINER_ENTRYPOINT, entrypoint)
    task_path = _task_yaml(tmp_path)

    result, docker_args, _ = _run_launcher(tmp_path, str(task_path), launcher=launcher)

    assert result.returncode == 0, result.stderr
    run_dir = Path(result.stdout.removeprefix("Run directory: ").strip())
    assert run_dir.parent == fake_repo / "datasets" / "autonomous_runs"
    assert run_dir.name.startswith("run.")
    assert run_dir.is_dir()
    assert f"type=bind,src={fake_repo},dst=/workspaces/isaac_autodata,readonly" in _option_values(
        docker_args, "--mount"
    )


def test_repository_root_is_rejected_as_writable_run_directory(tmp_path: Path) -> None:
    task_path = _task_yaml(tmp_path)

    result, docker_args, _ = _run_launcher(tmp_path, "--run-dir", str(REPO_ROOT), str(task_path))

    assert result.returncode == 2
    assert "run directory is too broad" in result.stderr
    assert docker_args == []


def test_existing_different_task_copy_is_not_overwritten(tmp_path: Path) -> None:
    task_path = _task_yaml(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    task_copy = run_dir / "task.yaml"
    existing_contents = b"schema_version: existing\n"
    task_copy.write_bytes(existing_contents)

    result, docker_args, _ = _run_launcher(
        tmp_path,
        "--run-dir",
        str(run_dir),
        str(task_path),
    )

    assert result.returncode == 2
    assert "run directory already contains task.yaml" in result.stderr
    assert docker_args == []
    assert task_copy.read_bytes() == existing_contents
    assert task_copy.stat().st_uid == os.getuid()


def test_explicit_headless_is_accepted(tmp_path: Path) -> None:
    task_path = _task_yaml(tmp_path)
    run_dir = tmp_path / "run"

    result, docker_args, _ = _run_launcher(
        tmp_path,
        "--headless",
        "--run-dir",
        str(run_dir),
        str(task_path),
    )

    assert result.returncode == 0, result.stderr
    assert "--gui" not in docker_args
    assert not any("/tmp/.X11-unix" in argument for argument in docker_args)


def test_gui_and_headless_conflict_before_docker_inspection(tmp_path: Path) -> None:
    task_path = _task_yaml(tmp_path)

    result, docker_args, inspect_calls = _run_launcher(
        tmp_path,
        "--gui",
        "--headless",
        str(task_path),
    )

    assert result.returncode == 2
    assert "conflicts" in result.stderr
    assert docker_args == []
    assert inspect_calls == []


def test_shell_scripts_have_valid_bash_syntax() -> None:
    for script in (LAUNCHER, CONTAINER_ENTRYPOINT):
        subprocess.run(["bash", "-n", str(script)], check=True, timeout=10)
