#!/bin/bash

echo "Running MFEM integration tests"

CLANG=$1
CLANGENZYME=$2
NPROC=$3

echo "$CLANG" "$CLANGENZYME" "$NPROC"

apt install -y openmpi-bin openmpi-common libopenmpi-dev libhypre-dev libmetis-dev

git clone -b dfem-dev --single-branch https://github.com/mfem/mfem.git
cd mfem
git apply --check ../integration/mfem/mfem.patch

# if [ -d "build" ]; then
#     rm -rf build
# else

echo $PWD
mkdir build
cd build
CXX=clang++-$CLANG cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_CXX_STANDARD_REQUIRED=ON \
-DMFEM_USE_MPI=ON \
-DHYPRE_INCLUDE_DIR=/usr/include/hypre \
-DMETIS_INCLUDE_DIR=/usr/include \
-DMFEM_USE_ENZYME=ON \
-DENZYME_DIR=$CLANGENZYME

echo $PWD
make -j $NPROC

echo $PWD
cd tests
make -C .. -j $NPROC punit_tests && ./unit/punit_tests "[dFEM]"
