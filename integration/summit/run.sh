#!/bin/bash

# set explicitly, the workflow invokes this as `bash run.sh`, which ignores the shebang
set -e

echo "Running sumMIT integration tests"

CLANGV=$1
ENZYME_DIR=$2
NPROC=${3:-`nproc`}
LLDENZYME=$ENZYME_DIR/LLDEnzyme-$CLANGV.so

# sumMIT lives in a private repository, so the workflow forwards SUMMIT_TOKEN, a
# token with read access to it. Fail early rather than on a bare git credential
# prompt, so an expired or missing token is obvious from the log.
if [ -z "$SUMMIT_URL" ] && [ -z "$SUMMIT_TOKEN" ]; then
    echo "SUMMIT_TOKEN is empty; cloning the private sumMIT repository needs a token" >&2
    echo "with read access to MIT-PSAAP-IV/sumMIT. If the secret exists, it has" >&2
    echo "most likely expired and needs to be reissued." >&2
    exit 1
fi
SUMMIT_URL=${SUMMIT_URL:-https://${SUMMIT_TOKEN:+x-access-token:$SUMMIT_TOKEN@}github.com/MIT-PSAAP-IV/sumMIT.git}
# the Enzyme tests live on the enzyme-ad branch; switch to main once it lands
SUMMIT_BRANCH=${SUMMIT_BRANCH:-enzyme-ad}

echo "CLANGV: $CLANGV"
echo "ENZYME_DIR: $ENZYME_DIR"
echo "LLDENZYME: $LLDENZYME"
echo "NPROC: $NPROC"
echo "SUMMIT_BRANCH: $SUMMIT_BRANCH"

if [ ! -f "$LLDENZYME" ]; then
    echo "LLDEnzyme plugin not found at $LLDENZYME" >&2
    exit 1
fi

WORKDIR=$PWD

echo "Installing dependencies"
apt-get update
# lld is required because sumMIT links the Enzyme tests through LLDEnzyme, the
# remaining packages are sumMIT's documented dependency set (doc/wiki/installing)
apt-get install -y \
    lld-$CLANGV gfortran cmake git ca-certificates \
    libgtest-dev libopenmpi-dev libmetis-dev libparmetis-dev \
    petsc-dev slepc-dev libhdf5-dev libhdf5-openmpi-dev \
    libeigen3-dev libgsl-dev libyaml-cpp-dev libjsoncpp-dev \
    libvtk9-dev python3-vtk9 qtbase5-dev qtchooser qt5-qmake qttools5-dev-tools \
    pybind11-dev libpython3-dev python3-pip

echo "Building pyre"
# pyre is not packaged for Ubuntu, so build it from source. The clone cannot be
# shallow because pyre derives its version with git describe, which needs tags.
git clone https://github.com/pyre/pyre.git
cmake -S pyre -B pyre-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/usr" \
    -DCMAKE_MODULE_PATH="$WORKDIR/pyre/.cmake" \
    -DCMAKE_INSTALL_PREFIX="$WORKDIR/pyre-install"
cmake --build pyre-build --target install -j $NPROC

echo "Cloning sumMIT"
git clone --depth 1 -b $SUMMIT_BRANCH "$SUMMIT_URL" sumMIT

echo "Writing sumMIT toolchain file"
# sumMIT resolves its dependencies through a toolchain file rather than through
# plain cmake variables; see doc/wiki/installing/cmake.md in the sumMIT tree
cat > $WORKDIR/enzyme_toolchain.cmake <<EOF
include(summit_init)

set(CMAKE_C_COMPILER "clang-$CLANGV" CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER "clang++-$CLANGV" CACHE FILEPATH "" FORCE)
set(CMAKE_Fortran_COMPILER "gfortran" CACHE FILEPATH "" FORCE)

set(SYS_PREFIX "/usr")

set(HDF5_PREFER_PARALLEL ON)
set(PYBIND11_FINDPYTHON ON)

summit_set_dependency(Eigen3   "\${SYS_PREFIX}"                              "Eigen3::Eigen")
summit_set_dependency(GTEST    "\${SYS_PREFIX}"                              "GTest::gtest")
summit_set_dependency(GSL      "\${SYS_PREFIX}"                              "GSL::gsl")
summit_set_dependency(HDF5     "\${SYS_PREFIX}"                              "HDF5::HDF5")
summit_set_dependency(METIS    "\${SYS_PREFIX}"                              "METIS::METIS")
summit_set_dependency(MPI      "\${SYS_PREFIX}/lib/x86_64-linux-gnu/openmpi" "MPI::MPI_CXX")
summit_set_dependency(ParMETIS "\${SYS_PREFIX}"                              "ParMETIS::ParMETIS")
summit_set_dependency(PETSc    "\${SYS_PREFIX}"                              "PETSc::PETSc")
summit_set_dependency(pybind11 "\${SYS_PREFIX}"                              "pybind11::module")
summit_set_dependency(PYRE     "$WORKDIR/pyre-install"                       "pyre::pyre")
summit_set_dependency(Python   "\${SYS_PREFIX}"                              "Python3::Python")
summit_set_dependency(SLEPc    "\${SYS_PREFIX}"                              "SLEPc::SLEPc")
summit_set_dependency(VTK      "\${SYS_PREFIX}"                              "VTK::CommonCore")
summit_set_dependency(VTK      "\${SYS_PREFIX}"                              "VTK::IOXML")
summit_set_dependency(VTK      "\${SYS_PREFIX}"                              "VTK::CommonDataModel")
summit_set_dependency(yaml-cpp "\${SYS_PREFIX}"                              "yaml-cpp")

list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
set(CMAKE_PREFIX_PATH "\${CMAKE_PREFIX_PATH}" CACHE STRING "" FORCE)
summit_generate_features_file()
EOF

echo "Configuring sumMIT"
cmake -S sumMIT -B summit-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$WORKDIR/enzyme_toolchain.cmake \
    -DSUMMIT_COMPILER_SET=CLANG \
    -DSUMMIT_BUILD_TESTING=ON \
    -DSUMMIT_DEBUG=OFF \
    -DWITH_ENZYME=ON \
    -DSUMMIT_ENZYME_LLD_PLUGIN=$LLDENZYME

# The Enzyme test drivers are EXCLUDE_FROM_ALL, and the ctest name of each one is
# its target name, so ask ctest which ones exist rather than hardcoding the list.
TARGETS=`ctest --test-dir summit-build -N -R enzyme | sed -n 's/^ *Test *#[0-9]*: *//p'`

if [ -z "$TARGETS" ]; then
    echo "No sumMIT Enzyme tests were registered" >&2
    exit 1
fi

echo "Building Enzyme test drivers:"
echo "$TARGETS"
cmake --build summit-build --target $TARGETS -j $NPROC

ctest --test-dir summit-build -R enzyme --output-on-failure
