#!/usr/bin/env bash
#
# Build a self-contained CellOracle distribution bundle.
#
#   ./scripts/make_bundle.sh [output_dir]
#
# Produces, in ./dist (or the given output_dir):
#
#   celloracle-<version>-py3-none-any.whl   installable wheel (code + small assets)
#   celloracle-<version>.tar.gz             source distribution
#   celloracle-data-<version>.tar.gz        optional offline dataset bundle
#
# The data bundle contains the tutorial / base-GRN / motif datasets that the
# wheel otherwise downloads on demand.  Extract it into ~/celloracle_data/
# (the default CELLORACLE_DATA_DIR) to run completely offline:
#
#   mkdir -p ~/celloracle_data
#   tar -xzf dist/celloracle-data-<version>.tar.gz -C ~/celloracle_data
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"
OUT="${1:-$HERE/dist}"
VERSION="$(sed -n "s/^__version__ *= *['\"]\(.*\)['\"].*/\1/p" celloracle/version.py | head -n 1)"
if [ -z "$VERSION" ]; then
    echo "could not determine version from celloracle/version.py" >&2
    exit 1
fi

mkdir -p "$OUT"

echo ">>> building wheel + sdist (version ${VERSION})"
"$PYTHON" -m build --outdir "$OUT"        # requires: pip install build

echo ">>> building offline data bundle"
DATA_DIRS=(anndata TFinfo_data promoter_base_GRN tutorial_data motif_data)
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
for d in "${DATA_DIRS[@]}"; do
    if [ -d "celloracle/data/$d" ]; then
        mkdir -p "$TMP/$d"
        cp -R "celloracle/data/$d/." "$TMP/$d/"
    fi
done
# Top-level directories (anndata/, TFinfo_data/, ...) so the archive can be
# extracted straight into CELLORACLE_DATA_DIR (~/celloracle_data).
tar -czf "$OUT/celloracle-data-${VERSION}.tar.gz" -C "$TMP" .

echo ">>> done"
ls -lh "$OUT"
