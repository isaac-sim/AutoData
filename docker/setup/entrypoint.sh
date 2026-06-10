#!/bin/bash
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
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

# The repo is bind-mounted over $WORKDIR at runtime, which hides the _isaac_sim symlink baked into
# the image. Recreate it (force, in case the host carries a stale symlink from a local conda setup)
# so Isaac Lab finds Isaac Sim at /isaac-sim.
ISAACLAB_DIR="$WORKDIR/submodules/IsaacLab-Arena/submodules/IsaacLab"
if [ -d "$ISAACLAB_DIR" ]; then
    ln -sfn /isaac-sim "$ISAACLAB_DIR/_isaac_sim"
fi

# Run the passed command as the dev user (bash -i so the python/pytest aliases expand), or drop into
# an interactive shell.
if [ $# -ge 1 ]; then
    exec sudo --preserve-env -u "$DOCKER_RUN_USER_NAME" \
        -- env HOME="/home/$DOCKER_RUN_USER_NAME" bash -ic "$*"
else
    exec su "$DOCKER_RUN_USER_NAME"
fi
