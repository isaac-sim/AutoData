#!/bin/bash
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Build Isaac AutoData Sphinx documentation.
#
#   ./scripts/ci/run_docs_build.sh verify   # PR/pre-merge: single-version build (-W)
#   ./scripts/ci/run_docs_build.sh publish  # Pages: multi-version build (-W)
#
# Published site: https://isaac-sim.github.io/Isaac-AutoData/
#
# verify builds the checked-out branch into docs/_build/current/html.
# publish runs sphinx-multiversion for whitelisted branches/tags (see docs/conf.py)
# and writes docs/_build/{main,v*,release/*}/. Nightly publish refreshes main/ on
# each run; versioned outputs are rebuilt from immutable refs and stay stable.

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
