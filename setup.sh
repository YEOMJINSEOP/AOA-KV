#!/bin/bash
# One-shot setup for a fresh box (Vast.ai etc).
#
#   conda env create -f environment.yml && conda activate aoakv
#   bash setup.sh                  # pip layer + nltk data        (accuracy ready)
#   bash setup.sh --build-kernels  # + both kernel trees compiled (latency ready)
#   bash setup.sh --check          # verify the install
#
# Flags:
#   --deps-only          pip layer + nltk data (default)
#   --build-kernels      also fetch third-party sources and compile both trees
#   --with-kernel-bench  additionally build the standalone nvbench micro-benchmarks
#   --check              run the self-check and exit
#   --skip-deps          with --build-kernels: don't touch pip
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
AOAKV_ROOT="$(pwd)"

DO_DEPS=1
DO_KERNELS=0
DO_KERNEL_BENCH=0
DO_CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --deps-only)         DO_KERNELS=0; shift ;;
        --build-kernels)     DO_KERNELS=1; shift ;;
        --with-kernel-bench) DO_KERNELS=1; DO_KERNEL_BENCH=1; shift ;;
        --skip-deps)         DO_DEPS=0; shift ;;
        --check)             DO_CHECK_ONLY=1; shift ;;
        -h|--help)           sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Pinned third-party revisions, taken from the upstream quest submodules. raft
# is not listed: both CMake projects pull it through rapids-cmake/CPM at
# configure time, so the checkout upstream carries is never used.
FLASHINFER_URL="https://github.com/flashinfer-ai/flashinfer"
FLASHINFER_SHA="9f49803b1db0a40ea0019ad98b8bb5d4f1593c77"
PYBIND_URL="https://github.com/pybind/pybind11"
PYBIND_SHA="768cebe17e65c2a0a64ed067510729efc3c7ff6c"
NVBENCH_URL="https://github.com/NVIDIA/nvbench"
NVBENCH_SHA="d8dced8a64d9ce305add92fa6d274fd49b569b7e"
GTEST_URL="https://github.com/google/googletest"
GTEST_SHA="6a5938233b6519ba99ddb7c7314d45d3fa877969"

TREES=(aoa_kv quest_orig)

log() { echo -e "\n=== $* ==="; }

# ---------------------------------------------------------------- self-check
self_check() {
    log "self-check"
    local fail=0

    python - <<'PY' || fail=1
import importlib, sys
sys.path.insert(0, ".")
missing = []
for m in ["torch", "transformers", "datasets", "numpy", "scipy", "nltk", "rouge",
          "jieba", "fuzzywuzzy", "wonderwords", "tenacity", "requests", "yaml",
          "flash_attn", "compat.manifest"]:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m} ({type(e).__name__})")
print("python deps:", "OK" if not missing else "MISSING -> " + ", ".join(missing))
try:
    import nltk; nltk.data.find("tokenizers/punkt")
    print("nltk punkt: OK")
except Exception:
    print("nltk punkt: MISSING (run: python -m nltk.downloader punkt punkt_tab)")
    missing.append("punkt")
try:
    import torch
    print(f"torch      : {torch.__version__}, cuda={torch.version.cuda}, "
          f"gpus={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"gpu 0      : {torch.cuda.get_device_name(0)} "
              f"(sm_{''.join(map(str, torch.cuda.get_device_capability(0)))})")
except Exception as e:
    print("torch      : FAILED", e)
    missing.append("torch")
sys.exit(1 if missing else 0)
PY

    # RULER needs its synthetic sources; they ship with the repo.
    for f in RULER/scripts/data/synthetic/json/{PaulGrahamEssays,squad,hotpotqa}.json; do
        [ -s "$f" ] && echo "data       : OK  $f" || { echo "data       : MISSING $f"; fail=1; }
    done

    # Kernel trees: compiled or not, and are they actually different binaries?
    for t in "${TREES[@]}"; do
        so=$(find "kernels/$t/quest" -maxdepth 1 -name "_kernels*.so" 2>/dev/null | head -1)
        if [ -n "$so" ]; then
            echo "kernel $t: built  md5=$(md5sum "$so" | cut -c1-8)"
        else
            echo "kernel $t: not built (bash setup.sh --build-kernels)"
        fi
    done

    # No machine-specific paths or secrets may creep back in.
    # (this file is excluded from both greps: it contains the patterns themselves)
    if grep -rn --exclude=setup.sh "WANDB_API_KEY=[\"']*[0-9a-zA-Z]" \
         --include="*.sh" --include="*.py" . >/dev/null 2>&1; then
        echo "secrets    : FAIL — a hardcoded WANDB_API_KEY is present"; fail=1
    else
        echo "secrets    : OK (no hardcoded keys)"
    fi
    # Only *live* code matters here; a few commented-out debug dumps inherited
    # from the research tree still mention old machine paths.
    bad_paths=$(grep -rn --exclude=setup.sh --exclude-dir=__pycache__ \
         "/NAS\|/SSD/JSY" --include="*.py" --include="*.sh" \
         evaluation filtering RULER scripts latency compat env.sh \
         | grep -v ":[0-9]*: *#" || true)
    if [ -n "$bad_paths" ]; then
        echo "paths      : FAIL — hardcoded /NAS or /SSD/JSY path in live code:"
        echo "$bad_paths" | head -5
        fail=1
    else
        echo "paths      : OK (no machine-specific paths in live code)"
    fi

    [ "$fail" = 0 ] && echo -e "\nself-check passed" || { echo -e "\nself-check FAILED"; return 1; }
}

if [ "$DO_CHECK_ONLY" = 1 ]; then
    self_check
    exit $?
fi

# ------------------------------------------------------------------ pip layer
if [ "$DO_DEPS" = 1 ]; then
    log "installing python dependencies"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    # flash-attn compiles against an already-installed torch, so it cannot go in
    # the same resolve as torch itself.
    python -m pip install flash-attn==2.6.3 --no-build-isolation
    log "downloading nltk tokenizer data"
    python -m nltk.downloader punkt punkt_tab
fi

# -------------------------------------------------------------- kernel builds
if [ "$DO_KERNELS" = 1 ]; then
    command -v nvcc >/dev/null || {
        echo "Error: nvcc not found. The kernel build needs the CUDA toolkit." >&2
        echo "       Accuracy runs do not need it — use --deps-only." >&2
        exit 1
    }

    # gcc-11 is what upstream used; fall back to the default compiler.
    if command -v gcc-11 >/dev/null; then
        HOST_CC=$(command -v gcc-11); HOST_CXX=$(command -v g++-11)
    else
        HOST_CC=$(command -v gcc); HOST_CXX=$(command -v g++)
    fi
    CUDA_ARCH="${AOAKV_CUDA_ARCH:-$(python -c "import torch;print(''.join(map(str,torch.cuda.get_device_capability(0))))" 2>/dev/null || echo native)}"
    TORCH_CMAKE=$(python -c "import torch;print(torch.utils.cmake_prefix_path)")
    log "kernel build: host cc=$HOST_CC  cuda arch=$CUDA_ARCH"

    fetch() {  # fetch <dir> <url> <sha>
        local dir=$1 url=$2 sha=$3
        if [ -d "$dir/.git" ]; then
            echo "  $dir already present"
            return
        fi
        echo "  cloning $url @ ${sha:0:8}"
        git init -q "$dir"
        git -C "$dir" remote add origin "$url" 2>/dev/null || true
        git -C "$dir" fetch -q --depth 1 origin "$sha"
        git -C "$dir" checkout -q FETCH_HEAD
    }

    log "fetching third-party sources (shared by both trees)"
    mkdir -p third_party
    fetch third_party/flashinfer "$FLASHINFER_URL" "$FLASHINFER_SHA"
    fetch third_party/pybind     "$PYBIND_URL"     "$PYBIND_SHA"
    if [ "$DO_KERNEL_BENCH" = 1 ]; then
        fetch third_party/nvbench    "$NVBENCH_URL" "$NVBENCH_SHA"
        fetch third_party/googletest "$GTEST_URL"   "$GTEST_SHA"
    fi

    for t in "${TREES[@]}"; do
        log "building kernels/$t"
        mkdir -p "kernels/$t/kernels/3rdparty"
        for dep in flashinfer pybind nvbench googletest; do
            if [ -d "third_party/$dep" ]; then
                ln -sfn "${AOAKV_ROOT}/third_party/$dep" "kernels/$t/kernels/3rdparty/$dep"
            fi
        done

        ( cd "kernels/$t/quest/ops"
          mkdir -p build && cd build
          cmake -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
                -DCMAKE_C_COMPILER="$HOST_CC" \
                -DCMAKE_CXX_COMPILER="$HOST_CXX" \
                -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
                -GNinja .. > cmake_configure.log
          ninja )

        # link the freshly built extension next to the package it belongs to
        so=$(find "kernels/$t/quest/ops/build" -maxdepth 1 -name "_kernels*.so" | head -1)
        [ -n "$so" ] || { echo "Error: build produced no _kernels*.so for $t" >&2; exit 1; }
        rm -f "kernels/$t/quest/"_kernels*.so
        ln -s "$(realpath "$so")" "kernels/$t/quest/"
        echo "  -> kernels/$t/quest/$(basename "$so")  md5=$(md5sum "$so" | cut -c1-8)"

        if [ "$DO_KERNEL_BENCH" = 1 ]; then
            ( cd "kernels/$t/kernels"
              mkdir -p build && cd build
              cmake -DCMAKE_C_COMPILER="$HOST_CC" \
                    -DCMAKE_CXX_COMPILER="$HOST_CXX" \
                    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
                    -GNinja .. > cmake_configure.log
              ninja )
        fi
    done

    # The whole point is two *different* binaries; say so explicitly.
    a=$(md5sum kernels/aoa_kv/quest/_kernels*.so | cut -d' ' -f1)
    b=$(md5sum kernels/quest_orig/quest/_kernels*.so | cut -d' ' -f1)
    if [ "$a" = "$b" ]; then
        echo "ERROR: both trees produced an identical binary ($a) — the AOA-KV diff did not compile in." >&2
        exit 1
    fi
    echo -e "\naoa_kv md5     = $a\nquest_orig md5 = $b"
    echo "Next: python latency/selftest_kernel.py --kernel aoa_kv   (and --kernel quest_orig)"
fi

self_check
