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
  echo "  --compiler <family> Specify the Fortran compiler family (intel or flang)."
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
  # rm -rf "${BUILD_DIR}/Fortran"
  # rm -rf "${BUILD_DIR}/test/Fortran"
else
  echo "Rebuilding..."
fi
mkdir -p "${BUILD_DIR}"

# Select compiler
CC=clang
CXX=clang++
ENZYME_FLANG=OFF
ENZYME_IFX=OFF
if [ "${COMPILER}" == "intel" ]; then
  # CC=icx
  # CXX=icx
  FC=ifx
  ENZYME_IFX=ON
elif [ "${COMPILER}" == "flang" ]; then
  FC=flang
  ENZYME_FLANG=ON
else
  echo "Unsupported compiler: ${COMPILER}"
  echo "Supported compilers are: intel or flang"
  exit 1
fi

# Build Enzyme
export SPACK_COLOR=false
LLVM_DIR="$(spack find --format '{prefix}' llvm)"
export SPACK_COLOR=always
# cd "${BUILD_DIR}"
# cmake ../../enzyme -G Ninja \
cmake -G Ninja -S enzyme -B "${BUILD_DIR}" \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_Fortran_COMPILER="${FC}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DLLVM_EXTERNAL_LIT="$(which lit)" \
  -DENZYME_FORTRAN=ON \
  -DENZYME_FLANG=${ENZYME_FLANG} \
  -DENZYME_IFX=${ENZYME_IFX} \
  -DENZYME_ENABLE_PLUGINS=ON
cd ${BUILD_DIR}
ninja
