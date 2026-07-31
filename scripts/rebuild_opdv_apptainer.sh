#!/usr/bin/env bash
# =============================================================================
# Rebuild the OPD-V Apptainer image.
#
# Fast rebuild with an existing environment.tar.gz:
#   bash "$0"
#
# Full rebuild with conda-pack:
#   bash "$0" --full-pack
#
# Output:
#   ${OPDV_APPTAINER_DIR}/opd-v.sif
#   Existing image is backed up as opd-v.sif.bak.<timestamp>.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-}"
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
ENV_NAME="${ENV_NAME:-opd-v}"
OUT_DIR="${OPDV_APPTAINER_DIR:-${PROJECT_ROOT}/../apptainer}"
OUT_SIF="${OUT_DIR}/opd-v.sif"
BASE_DOCKER_IMAGE="${BASE_DOCKER_IMAGE:-quay.io/rockylinux/rockylinux:9}"
# 已有 tarball（上次 conda-pack 产物，可复用）
DEFAULT_TARBALL="${OUT_DIR}/environment.tar.gz"

FULL_PACK=0
TARBALL="${DEFAULT_TARBALL}"

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  sed -n '1,20p' "$0" | sed 's/^# \{0,1\}//'
  echo ""
  echo "选项:"
  echo "  --full-pack      重新 conda-pack（慢）；默认复用已有 tarball"
  echo "  --tarball PATH   指定 environment.tar.gz"
  echo "  --out PATH       输出 .sif（默认: ${OUT_SIF}）"
  echo "  -h, --help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-pack) FULL_PACK=1; shift ;;
    --tarball) TARBALL="$2"; shift 2 ;;
    --out) OUT_SIF="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数: $1" ;;
  esac
done

APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
[[ -n "${APPTAINER_BIN}" ]] || die "未找到 apptainer"

STAMP="$(date +%Y%m%d_%H%M%S)"
WORKDIR="${OUT_DIR}/rebuild_opd-v_${STAMP}"
mkdir -p "$WORKDIR"

export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/opdv_apptainer_build_${STAMP}}"
export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"
export TMPDIR="${APPTAINER_TMPDIR}"
mkdir -p "$APPTAINER_TMPDIR"

if [[ "$FULL_PACK" -eq 1 ]]; then
  TARBALL="${WORKDIR}/environment.tar.gz"
  if [[ -n "${CONDA_ROOT}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  elif [[ -n "${CONDA_EXE}" ]]; then
    CONDA_BASE="$("${CONDA_EXE}" info --base)"
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
  else
    die "Set CONDA_ROOT or make conda available in PATH"
  fi
  conda activate base
  command -v conda-pack >/dev/null 2>&1 || die "需要: conda install -n base -c conda-forge conda-pack"
  echo ">>> conda-pack -n ${ENV_NAME} ..."
  conda pack -n "$ENV_NAME" -o "$TARBALL" --ignore-editable-packages --ignore-missing-files
else
  [[ -f "$TARBALL" ]] || die "找不到 tarball: $TARBALL（用 --full-pack 重新打包）"
  echo ">>> 复用 tarball: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
fi

DEF_FILE="${WORKDIR}/opd-v.def"
STAGE_IN_BUILD="${WORKDIR}/environment.tar.gz"
if [[ "$TARBALL" != "$STAGE_IN_BUILD" ]]; then
  echo ">>> 复制 tarball 到构建目录 ..."
  cp -f "$TARBALL" "$STAGE_IN_BUILD"
fi

cat > "$DEF_FILE" <<'DEFEOF'
Bootstrap: docker
From: BASE_IMAGE_PLACEHOLDER

%files
    STAGE_PATH_PLACEHOLDER /opt/conda-pack-environment.tar.gz

%post
    set -e
    STAGE=/opt/conda-pack-environment.tar.gz

    # 基础工具 + vLLM/Triton JIT 编译所需的头文件与 gcc
    if command -v dnf >/dev/null 2>&1; then
        dnf install -y --setopt=install_weak_deps=False \
            tar gzip findutils which \
            gcc gcc-c++ make \
            glibc-devel glibc-headers kernel-headers \
            binutils
        dnf clean all || true
    fi

    # 验证编译环境（构建失败时便于排查）
    test -f /usr/include/stdlib.h || { echo "FATAL: stdlib.h missing"; exit 1; }
    gcc --version | head -1 || { echo "FATAL: gcc missing"; exit 1; }

    test -f "$STAGE" || { echo "FATAL: missing $STAGE"; exit 1; }
    mkdir -p /opt/conda-env
    tar -xzf "$STAGE" -C /opt/conda-env
    rm -f "$STAGE"
    cd /opt/conda-env
    if [ -f ./bin/conda-unpack ]; then
        ./bin/python ./bin/conda-unpack
    fi

%environment
    export PATH=/opt/conda-env/bin:$PATH
    export LD_LIBRARY_PATH=/opt/conda-env/lib

%labels
    PackedEnv opd-v
    BuildNote gcc-for-vllm-triton
DEFEOF

sed -i "s|BASE_IMAGE_PLACEHOLDER|${BASE_DOCKER_IMAGE}|" "$DEF_FILE"
sed -i "s|STAGE_PATH_PLACEHOLDER|${STAGE_IN_BUILD}|" "$DEF_FILE"

# 备份旧镜像
if [[ -f "$OUT_SIF" ]]; then
  BAK="${OUT_SIF}.bak.${STAMP}"
  echo ">>> 备份旧镜像 -> $BAK"
  mv -f "$OUT_SIF" "$BAK"
fi

echo ">>> DEF: $DEF_FILE"
echo ">>> 构建 -> $OUT_SIF"
echo ">>> APPTAINER_TMPDIR=$APPTAINER_TMPDIR"

if "$APPTAINER_BIN" build --fakeroot "$OUT_SIF" "$DEF_FILE"; then
  echo ">>> OK: $OUT_SIF ($(du -h "$OUT_SIF" | cut -f1))"
elif "$APPTAINER_BIN" build "$OUT_SIF" "$DEF_FILE"; then
  echo ">>> OK (no fakeroot): $OUT_SIF"
else
  die "apptainer build 失败"
fi

rm -rf "$APPTAINER_TMPDIR"

echo ""
echo ">>> 验证（登录节点，无 GPU 时可只测 import）："
echo "    apptainer exec $OUT_SIF bash -lc 'gcc --version; test -f /usr/include/stdlib.h; python -V'"
echo "    apptainer exec --nv $OUT_SIF bash -lc 'export LD_LIBRARY_PATH=/opt/conda-env/lib:\$LD_LIBRARY_PATH; python -c \"import torch; import vllm; print(torch.__version__, vllm.__version__)\"'"
echo ""
echo ">>> Training:"
echo "    Use the image with your local launcher or scheduler."
