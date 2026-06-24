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
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap | sed -n '2s/\.//p')

    echo "Updating system and installing dependencies"
    apt update && apt install -y wget gnupg2 curl g++ freeglut3-dev libxmu-dev libxi-dev

    echo "Setting up NVIDIA repository pinning"
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

    echo "Fetching repository keys"
    # Fetches the latest keyring to ensure proper package verification
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

    echo "Adding CUDA network repository"
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" -y

    echo "Updating package lists"
    apt update

    echo "Installing CUDA Toolkit and Runtime"
    # Installs the full toolkit including runtime libraries
    apt install -y cuda-toolkit
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
    -DCMAKE_CUDA_COMPILER=clang++-$CLANGV
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
