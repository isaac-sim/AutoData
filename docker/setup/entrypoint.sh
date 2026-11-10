#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Container entrypoint. Creates a user inside the container matching the host UID/GID (passed via
# env by run_docker.sh) so files written into the bind-mounted repo keep host ownership, then runs
# the requested command (or an interactive shell) as that user.

# Exit on error
set -euo pipefail

# Resolve dynamic-linker cache (works around an occasional OpenCV shared-lib lookup issue).
ldconfig

# Recreate the host group/user inside the container.
groupadd --force --gid "$DOCKER_RUN_GROUP_ID" "$DOCKER_RUN_GROUP_NAME"
userdel "$DOCKER_RUN_USER_NAME" 2>/dev/null || true
userdel ubuntu 2>/dev/null || true
useradd --no-log-init \
        --uid "$DOCKER_RUN_USER_ID" \
        --gid "$DOCKER_RUN_GROUP_NAME" \
        --groups sudo,isaac-sim \
        --shell /bin/bash \
        "$DOCKER_RUN_USER_NAME"
chown "$DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME" "/home/$DOCKER_RUN_USER_NAME"
chown "$DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME" "$WORKDIR"

# Allow sudo without password
echo 'root:root' | chpasswd
echo "$DOCKER_RUN_USER_NAME:root" | chpasswd
echo "$DOCKER_RUN_USER_NAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
touch "/home/$DOCKER_RUN_USER_NAME/.sudo_as_admin_successful"

cp /etc/bash.bashrc "/home/$DOCKER_RUN_USER_NAME/.bashrc"
chown "$DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME" "/home/$DOCKER_RUN_USER_NAME/.bashrc"

# A binary Sim link makes Isaac Lab reject the locked wheel environment.
ISAACLAB_DIR="$WORKDIR/submodules/IsaacLab-Arena/submodules/IsaacLab"
if [ -e "$ISAACLAB_DIR/_isaac_sim" ] || [ -L "$ISAACLAB_DIR/_isaac_sim" ]; then
    echo "error: $ISAACLAB_DIR/_isaac_sim selects a binary Isaac Sim installation." >&2
    echo "Remove the link from the host checkout before using the locked Docker environment." >&2
    exit 1
fi

# Run the command as the dev user with the image's shell environment.
if [ $# -ge 1 ]; then
    # Quote each argument before passing it to the interactive shell so shell
    # preserves paths, marker expressions, and Python code.
    printf -v AUTODATA_COMMAND '%q ' "$@"
    exec sudo --preserve-env -u "$DOCKER_RUN_USER_NAME" \
        -- env HOME="/home/$DOCKER_RUN_USER_NAME" bash -ic "${AUTODATA_COMMAND}"
else
    exec su "$DOCKER_RUN_USER_NAME"
fi
