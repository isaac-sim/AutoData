#!/bin/bash
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Build Isaac AutoData Sphinx documentation.
#
#   ./scripts/ci/run_docs_build.sh verify
#   ./scripts/ci/run_docs_build.sh publish

set -euo pipefail

MODE="${1:-verify}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCS_DIR="${REPO_ROOT}/docs"

cd "${DOCS_DIR}"

pip install --no-cache-dir -r requirements.txt

case "${MODE}" in
  verify)
    make html SPHINXOPTS=-W
    ;;
  publish)
    make multi-docs SPHINXOPTS=-W
    touch _build/.nojekyll
    ;;
  *)
    echo "Usage: $0 {verify|publish}" >&2
    exit 1
    ;;
esac
