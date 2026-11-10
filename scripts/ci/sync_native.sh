#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root=${NATIVE_CACHE_ROOT:?}
key=${NATIVE_CACHE_KEY:?}
key_pattern='^uv[0-9.]+-python[0-9.]+-cuda[0-9.]+-[0-9._]+-[0-9a-f]{64}$'
[[ $key =~ $key_pattern ]]
[[ ! -L "$root" && ! -L "$root/.lock" && ! -L "$root/keys" ]]
mkdir -p -- "$root"

# The installer inherits this descriptor, keeping the lock if only the shell is killed.
exec 9>>"$root/.lock"
flock 9
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p -- "$root/keys"
current="$root/keys/$key"
[[ ! -L "$current" ]]
previous=
caches=()
for cache in "$root/keys/"*; do
    [[ -d "$cache" && ! -L "$cache" && ${cache##*/} =~ $key_pattern ]] || continue
    caches+=("$cache")
    if [[ "$cache" != "$current" && -f "$cache/.last-success" ]]; then
        if [[ -z "$previous" || "$cache/.last-success" -nt "$previous/.last-success" ]]; then
            previous=$cache
        fi
    fi
done

for cache in "${caches[@]}"; do
    if [[ "$cache" != "$current" && "$cache" != "$previous" ]]; then
        rm -rf -- "$cache"
    fi
done

mkdir -p -- "$current"
# Failed or interrupted retries must not count as a successful cache use.
rm -f -- "$current/.last-success"
UV_CACHE_DIR="$current" UV_LINK_MODE=copy uv sync --locked --group curobo
touch -- "$current/.last-success"
