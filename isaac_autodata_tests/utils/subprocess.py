# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import signal
import subprocess
import sys
import tempfile

_SUBPROCESS_TIMEOUT_SEC = int(os.environ.get("ISAAC_AUTODATA_SUBPROCESS_TIMEOUT", "1200"))


def run_subprocess(cmd: list[str], env: dict | None = None, timeout_sec: int | None = None) -> None:
    """Run a command in a subprocess with timeout.

    The child is launched in its own session/process group.

    Args:
        cmd: Command and arguments to execute.
        env: Environment for the child. Defaults to inheriting the current environment.
        timeout_sec: Wall-clock timeout in seconds. Defaults to
            ``_SUBPROCESS_TIMEOUT_SEC`` (env ``ISAAC_AUTODATA_SUBPROCESS_TIMEOUT``, fallback 1200).

    Raises:
        subprocess.TimeoutExpired: If the child does not finish within the timeout.
        subprocess.CalledProcessError: If the child exits with a non-zero return code.
    """
    if timeout_sec is None:
        timeout_sec = _SUBPROCESS_TIMEOUT_SEC
    if env is None:
        env = os.environ.copy()

    print(f"Running command (timeout={timeout_sec}s): {' '.join(cmd)}", flush=True)

    process = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        return_code = process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        sys.stderr.write(f"\n[isaac-autodata] Subprocess timed out after {timeout_sec}s\n")
        sys.stderr.flush()
        raise
    except BaseException:
        _kill_process_group(process)
        raise

    print(f"Command completed with return code: {return_code}", flush=True)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_subprocess_capture(cmd: list[str], env: dict | None = None, timeout_sec: int | None = None) -> str:
    """Run a command like :func:`run_subprocess` but capture and return its combined stdout+stderr.

    Output is streamed to a temporary file (avoiding pipe-buffer deadlocks on chatty children like
    Isaac Sim) and read back once the child exits. Same process-group / timeout semantics as
    :func:`run_subprocess`.

    Args:
        cmd: Command and arguments to execute.
        env: Environment for the child. Defaults to inheriting the current environment.
        timeout_sec: Wall-clock timeout in seconds. Defaults to ``_SUBPROCESS_TIMEOUT_SEC``.

    Returns:
        The child's combined stdout+stderr, decoded as UTF-8.

    Raises:
        subprocess.TimeoutExpired: If the child does not finish within the timeout.
        subprocess.CalledProcessError: If the child exits non-zero (captured output on ``.output``).
    """
    if timeout_sec is None:
        timeout_sec = _SUBPROCESS_TIMEOUT_SEC
    if env is None:
        env = os.environ.copy()

    print(f"Running command (timeout={timeout_sec}s): {' '.join(cmd)}", flush=True)

    with tempfile.TemporaryFile() as out_file:
        process = subprocess.Popen(cmd, env=env, start_new_session=True, stdout=out_file, stderr=subprocess.STDOUT)
        try:
            return_code = process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            sys.stderr.write(f"\n[isaac-autodata] Subprocess timed out after {timeout_sec}s\n")
            sys.stderr.flush()
            raise
        except BaseException:
            _kill_process_group(process)
            raise
        out_file.seek(0)
        output = out_file.read().decode("utf-8", errors="replace")

    print(f"Command completed with return code: {return_code}", flush=True)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd, output=output)
    return output


def _kill_process_group(process: subprocess.Popen) -> None:
    """SIGKILL the child's entire process group, then reap the direct child."""

    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=30)
