#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

# Function to display help text
show_help() {
  echo "Usage: $0 [--debug | -d] [--fresh | -f] [--std <standard>]"
  echo "                  [--compiler <compiler>] [--help | -h]"
  echo
  echo "Options:"
  echo "  --debug   | -d      Run in Debug mode."
  echo "  --compiler <family> Specify the compiler family"
  echo "                      (gnu/intel/nvidia/flang)."
  echo "  --help    | -h      Show this help message and exit."
}

# Check if a Python virtual environment is active
if [ -z "${VIRTUAL_ENV}" ]; then
  echo "No virtual environment is active. Please activate a virtual environment \
before running this script."
  exit 1
fi

# Parse command line arguments
BUILD_DIR="$(pwd)/build"
BUILD_TYPE=Release
COMPILER=flang
HELP=false
for arg in "$@"; do
  case $arg in
  --debug | -d)
    BUILD_DIR="${BUILD_DIR}_debug"
    BUILD_TYPE=Debug
    shift
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

# Run Fortran tests
cd "${BUILD_DIR}" || exit
ninja check-enzyme-fortran
