#!/usr/bin/env bash
set -euo pipefail

readonly RELEASE_TAG="activation-data-v1"
readonly ARCHIVE_NAME="Shapley-Moe-activations-v1.tar.gz"
readonly DEFAULT_URL="https://github.com/Alizen-1009/Shapley-Moe/releases/download/${RELEASE_TAG}/${ARCHIVE_NAME}"
readonly EXPECTED_SHA256="c0b114bf7d57b1e305673145ed6a4c9098d4f2c5cbbf7c2066eae74264e63bab"
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

url="${SHAPE_ACTIVATIONS_URL:-$DEFAULT_URL}"
destination="${SHAPE_ACTIVATIONS_DEST:-$REPO_ROOT}"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/shape-activations.XXXXXX")"
archive="${tmp_dir}/${ARCHIVE_NAME}"
trap 'rm -rf "$tmp_dir"' EXIT

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required" >&2
  exit 1
fi

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "error: sha256sum or shasum is required" >&2
    return 1
  fi
}

echo "Downloading precomputed activation data..."
curl --fail --location --retry 3 --output "$archive" "$url"

actual_sha256="$(sha256 "$archive")"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "error: checksum mismatch" >&2
  echo "expected: $EXPECTED_SHA256" >&2
  echo "actual:   $actual_sha256" >&2
  exit 1
fi

mkdir -p "$destination"
tar -xzf "$archive" -C "$destination"
echo "Activation data extracted under ${destination}/results/"
