#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

# Function to display help text
show_help() {
  echo "Usage: $0 [--debug | -d] [--fresh | -f] [--std <standard>]"
  echo "                  [--compiler <compiler>] [--help | -h]"
  echo
  echo "Options:"
  echo "  --debug   | -d      Compile in Debug mode."
  echo "  --fresh   | -f      Create a fresh build before compiling."
  echo "  --std <standard>    Specify the Fortran standard (e.g., f2008, f2018)."
  echo "  --compiler <family> Specify the compiler family"
  echo "                      (gnu/intel/nvidia/flang)."
  echo "  --help    | -h      Show this help message and exit."
}

# Parse command line arguments
BUILD_DIR="$(pwd)/build"
BUILD_TYPE=Release
FRESH_BUILD=false
FORTRAN_STANDARD=f2018
COMPILER=flang
HELP=false
for arg in "$@"; do
  case $arg in
  --debug | -d)
    BUILD_DIR="${BUILD_DIR}_debug"
    BUILD_TYPE=Debug
    shift
    ;;
  --fresh | -f)
    FRESH_BUILD=true
    shift
    ;;
  --std)
    FORTRAN_STANDARD="$2"
    shift 2
    ;;
  --compiler)
    COMPILER="$2"
    shift 2
    ;;
  --help | -h)
    HELP=true
    shift
    ;;
  *) ;;
  esac
done
BUILD_DIR="${BUILD_DIR}/${COMPILER}"

# Check for --help option
if [ "${HELP}" = true ]; then
  show_help
  exit 0
fi

# Check if a fresh build is requested
if [ "${FRESH_BUILD}" = true ]; then
  echo "Creating a fresh build..."
  rm -rf "${BUILD_DIR}"
else
  echo "Rebuilding..."
fi
mkdir -p "${BUILD_DIR}"

# Select compiler family
# TODO: Try lfortran
if [ "${COMPILER}" == "gnu" ]; then
  CC=gcc
  CXX=g++
  FC=gfortran
elif [ "${COMPILER}" == "intel" ]; then
  CC=icx-cc
  CXX=icx-cl
  FC=ifx
elif [ "${COMPILER}" == "nvidia" ]; then
  CC=nvcc
  CXX=nvc++
  FC=nvfortran
elif [ "${COMPILER}" == "flang" ]; then
  CC=clang
  CXX=clang++
  FC=flang
else
  echo "Unsupported compiler: ${COMPILER}"
  echo "Supported compilers are: gnu, intel, nvidia, flang"
  exit 1
fi

# Build Enzyme
cmake -G Ninja -S enzyme -B "${BUILD_DIR}" \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_Fortran_COMPILER="${FC}" \
  -DLLVM_DIR=/home/joe/tools/spack/opt/spack/linux-skylake/llvm-21.1.4-jtiynly3pgjolwfskf7iyjwi43vuoe5i/ \
  -DLLVM_EXTERNAL_LIT=/home/joe/.virtualenvs/enzyme/bin/lit \
  -DENZYME_IFX=ON
cd ${BUILD_DIR}
ninja
