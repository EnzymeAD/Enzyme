#!/bin/bash -e

echo "Running GridKit integration tests"

CLANGV=$1
ENZYME_DIR=$2

echo "CLANGV: $CLANGV" 
echo "ENZYME_DIR: $ENZYME_DIR"

git clone https://github.com/ORNL/GridKit.git
mkdir GridKit/build
cd GridKit/build
git checkout ea96fba7bcc9c9cc402eafd7ec3311bb08889598

cmake \
  -DCMAKE_CXX_COMPILER=clang++-$CLANGV \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGRIDKIT_ENABLE_ENZYME=On \
  -DENZYME_DIR=${ENZYME_DIR} \
  ..

make -j
make test
