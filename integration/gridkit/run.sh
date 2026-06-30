#!/bin/bash -e

echo "Running GridKit integration tests"

CLANGV=$1
ENZYME_DIR=$2

echo "CLANGV: $CLANGV" 
echo "ENZYME_DIR: $ENZYME_DIR"

git clone git@github.com:ORNL/GridKit.git
mkdir GridKit/build
cd GridKit/build
git checkout develop

cmake \
  -DCMAKE_CXX_COMPILER=clang++-$CLANGV \
  -DGRIDKIT_ENABLE_ENZYME=On \
  -DENZYME_DIR=${ENZYME_DIR} \
  ..

make -j
make test
