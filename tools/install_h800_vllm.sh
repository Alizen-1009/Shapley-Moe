#!/usr/bin/env bash
# Install the pinned SHAPE serving stack into an EXISTING Python 3.12 environment.
# Run on Linux x86_64. Does not create environments, launch services or modify models.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREFIX=${SHAPE_VLLM_ENV:-$HOME/miniconda3/envs/shape-vllm}
WORK=${SHAPE_WORKSPACE:-$HOME/shape-experiments}
PY="$PREFIX/bin/python"
"$PY" -c 'import sys,platform; assert sys.version_info[:2] == (3,12); assert platform.system() == "Linux" and platform.machine() == "x86_64"'
mkdir -p "$WORK"/{artifacts,src,logs,tmp,cache/pip}
export TMPDIR="$WORK/tmp" PIP_CACHE_DIR="$WORK/cache/pip"
# Obtain only the explicitly pinned archives. Reject checksum mismatches.
"$PY" - "$ROOT/configs/runtime/h800-vllm.json" "$WORK" <<'PY'
import hashlib,json,pathlib,sys,urllib.request
cfg=json.loads(pathlib.Path(sys.argv[1]).read_text()); work=pathlib.Path(sys.argv[2])
files=[(cfg['source']['archive_url'],'shape-vllm-4986caaa.tar.gz',cfg['source']['archive_sha256']),
       (cfg['precompiled']['wheel_url'],'vllm-0.23.1rc1.dev1458+ge222c33f2-cp38-abi3-manylinux_2_28_x86_64.whl',cfg['precompiled']['wheel_sha256'])]
for url,name,expected in files:
    path=work/'artifacts'/name
    if not path.exists():
        partial=path.with_name(path.name+'.part')
        with urllib.request.urlopen(url,timeout=30) as response, partial.open('wb') as target:
            while chunk:=response.read(1024*1024): target.write(chunk)
        assert hashlib.file_digest(partial.open('rb'),'sha256').hexdigest()==expected, name+' checksum mismatch'
        partial.replace(path)
    with path.open('rb') as source: actual=hashlib.file_digest(source,'sha256').hexdigest()
    assert actual==expected, name+' checksum mismatch'
    print('Verified',path,flush=True)
# Use the already-verified local wheel; preserve all original dependency hashes.
lock=pathlib.Path(sys.argv[1]).resolve().parents[2]/cfg['precompiled']['dependency_lock']
text=lock.read_text().replace(cfg['precompiled']['wheel_url'],(work/'artifacts'/files[1][1]).resolve().as_uri())
(work/'artifacts/runtime-install.lock.txt').write_text(text)
PY
SOURCE="$WORK/src/vllm-4986caaa"
if [[ ! -d "$SOURCE" ]]; then
  mkdir -p "$SOURCE"
  tar -xzf "$WORK/artifacts/shape-vllm-4986caaa.tar.gz" -C "$SOURCE" --strip-components=1
fi
# Mirror URLs are explicitly hash-pinned; three missing mirror wheels use PyPI.
# If PyPI downloads are slow, prefetch these exact files with their SHA256 hashes.
"$PY" -m pip install --require-hashes --only-binary=:all: \
  --report "$WORK/artifacts/runtime-install-report.json" \
  -r "$WORK/artifacts/runtime-install.lock.txt"
"$PY" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r "$ROOT/requirements/h800-vllm-build.txt"
export VLLM_USE_PRECOMPILED=1 VLLM_MAIN_CUDA_VERSION=13.0
export VLLM_VERSION_OVERRIDE=0.23.1rc1.dev1458+g4986caaa.precompiled
export VLLM_PRECOMPILED_WHEEL_LOCATION="$WORK/artifacts/vllm-0.23.1rc1.dev1458+ge222c33f2-cp38-abi3-manylinux_2_28_x86_64.whl"
"$PY" -m pip install --no-build-isolation --no-deps -e "$SOURCE"
"$PY" -m pip check
"$PY" -m pip freeze > "$WORK/artifacts/pip-freeze-installed.txt"
echo 'Installation complete. Model serving, routing equivalence and performance require separate validation.'
