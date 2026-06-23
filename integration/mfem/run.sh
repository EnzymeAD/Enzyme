#!/bin/bash

echo "Running MFEM integration tests"

CLANGV=$1
CLANGENZYME=$2
NPROC=$3

echo "$CLANGV" "$CLANGENZYME" "$NPROC"

nvidia-smi >/dev/null
USE_CUDA=0
COMPUTE_CAP=0
if [[ "$?" -eq 0 ]]; then
    USE_CUDA=1
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap | sed -n '2s/\.//p')
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
    -DCUDA_DIR=/usr/local/cuda \
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
