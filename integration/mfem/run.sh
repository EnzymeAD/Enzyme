#!/bin/bash

echo "Running MFEM integration tests"

CLANGV=$1
CLANGENZYME=$2
NPROC=$3

echo "$CLANGV" "$CLANGENZYME" "$NPROC"

USE_CUDA=0
COMPUTE_CAP=0
if nvidia-smi &> /dev/null; then
    echo "Using CUDA"
    USE_CUDA=1
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')

    if [ -z "$COMPUTE_CAP" ] || [ "$COMPUTE_CAP" = "0" ]; then
        COMPUTE_CAP=80
    fi

    echo "Updating system and installing dependencies"
    apt update && apt install -y wget curl g++ freeglut3-dev libxmu-dev libxi-dev

    echo "Detecting Ubuntu version and architecture"
    UBUNTU_VERSION="$(. /etc/os-release && echo "${VERSION_ID//./}")"
    DEB_ARCH="$(dpkg --print-architecture)"

    case "$DEB_ARCH" in
        amd64)
            NVIDIA_ARCH="x86_64"
            ;;
        arm64)
            NVIDIA_ARCH="sbsa"
            ;;
        *)
            echo "Unsupported architecture for CUDA repo: $DEB_ARCH" >&2
            exit 1
            ;;
    esac

    CUDA_REPO_BASE="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/${NVIDIA_ARCH}"

    echo "Using CUDA repo: $CUDA_REPO_BASE"

    echo "Setting up NVIDIA repository pinning"
    wget "${CUDA_REPO_BASE}/cuda-ubuntu${UBUNTU_VERSION}.pin"
    mv "cuda-ubuntu${UBUNTU_VERSION}.pin" /etc/apt/preferences.d/cuda-repository-pin-600

    echo "Installing NVIDIA CUDA keyring"
    wget "${CUDA_REPO_BASE}/cuda-keyring_1.1-1_all.deb"
    dpkg -i cuda-keyring_1.1-1_all.deb

    echo "Updating package lists"
    apt update

    echo "Installing CUDA Compiler and Libraries"
    apt install -y cuda-compiler-12-9 cuda-libraries-12-9 cuda-libraries-dev-12-9
fi

apt install -y openmpi-bin openmpi-common libopenmpi-dev libhypre-dev libmetis-dev

git clone -b dfem-dev --single-branch https://github.com/mfem/mfem.git
cd mfem

echo $PWD
mkdir build
cd build
if [[ "$USE_CUDA" -eq 0 ]]; then
    CXX=clang++-$CLANGV cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DMFEM_USE_MPI=ON \
    -DHYPRE_INCLUDE_DIR=/usr/include/hypre \
    -DMETIS_INCLUDE_DIR=/usr/include \
    -DMFEM_USE_ENZYME=ON \
    -DENZYME_DIR=$CLANGENZYME
else
    CXX=clang++-$CLANGV cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DMFEM_USE_MPI=ON \
    -DHYPRE_INCLUDE_DIR=/usr/include/hypre \
    -DMETIS_INCLUDE_DIR=/usr/include \
    -DMFEM_USE_ENZYME=ON \
    -DENZYME_DIR=$CLANGENZYME \
    -DMFEM_USE_CUDA=ON \
    -DCUDAToolkit_ROOT=/usr/local/cuda \
    -DCUDA_ARCH=$COMPUTE_CAP \
    -DCMAKE_CUDA_ARCHITECTURES="$COMPUTE_CAP" \
    -DCMAKE_CUDA_COMPILER=clang++-$CLANGV \
    -DCMAKE_CUDA_FLAGS="-Wno-unknown-cuda-version"
fi

echo $PWD
make -j $NPROC

echo $PWD
cd tests
if [[ "$USE_CUDA" -eq 0 ]]; then
    make -C .. -j $NPROC punit_tests && ./unit/punit_tests "[dFEM]"
else
    make -C .. -j $NPROC punit_tests && ./unit/punit_tests "[dFEM][GPU]"
fi
