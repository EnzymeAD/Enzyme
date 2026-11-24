#!/bin/bash
#
# Differentiates examples shipped with the OpenAD/F tool with Enzyme and checks
# each gradient against finite differences of the primal.
#
#   run.sh LLVM_VERSION ENZYME_DIR [NPROC]
#
# ENZYME_DIR is the Enzyme directory of a build configured with
# -DENZYME_FORTRAN=ON -DENZYME_FLANG=ON, e.g. build/Enzyme, with the
# FlangEnzyme, LLDEnzyme and EnzymeFortran targets built. FC overrides the
# flang-LLVM_VERSION default; ld.lld must be on the PATH.

set -euo pipefail

LLVMV=$1
ENZYME_DIR=$(realpath "$2")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

FC=${FC:-flang-$LLVMV}
ENZYME_MODULES=$ENZYME_DIR/../modules

echo "Running OpenAD integration tests"
echo "FC: $FC"
echo "ENZYME_DIR: $ENZYME_DIR"

OPENAD_URL=http://www.mcs.anl.gov/~utke/OpenAD_tars/493/OpenAD_2014-03-15.tgz
OPENAD_SHA256=1e41bc4388e3c7f99d64dde11e22e801d6705a0c6049b30c04b48f89d8b28800

WORK=$PWD/openad-work
rm -rf "$WORK"
mkdir -p "$WORK"
cd "$WORK"

curl -fsSL --retry 3 -o OpenAD.tgz "$OPENAD_URL"
echo "$OPENAD_SHA256  OpenAD.tgz" | sha256sum -c -
tar -xzf OpenAD.tgz OpenAD/Examples/BoxModel OpenAD/Examples/FDC \
  OpenAD/Examples/ShallowWater
EXAMPLES=$WORK/OpenAD/Examples

# Examples Enzyme cannot differentiate yet. They still run, and the script
# fails if one of them starts to pass so that it is moved out of this list.
#   ShallowWater: type analysis cannot deduce the type of whole-array copies
#   between module arrays (u, eta), "Cannot deduce type of copy".
EXPECTED_FAILURES=" ${OPENAD_EXPECTED_FAILURES-ShallowWater} "

PASSED=()
FAILED=()

# build_and_run NAME FFLAGS LIBS SOURCE...
#
# Builds the sources (modules first) with flang, Enzyme loaded through
# FlangEnzyme's config file, and full LTO, so that Enzyme differentiates at link
# time with every callee in view. Runs the result in a fresh directory NAME,
# where the program checks its gradient and exits nonzero on a mismatch.
build_and_run() {
  local name=$1 fflags=$2 libs=$3
  shift 3
  local dir=$WORK/$name
  mkdir -p "$dir"
  echo "=== $name"
  # shellcheck disable=SC2086
  if "$FC" --config="$ENZYME_DIR/FlangEnzyme-$LLVMV.cfg" -flto -O2 $fflags \
    -I"$ENZYME_MODULES" -module-dir "$dir" \
    -Wl,-mllvm,-enzyme-global-activity "$@" -o "$dir/$name" $libs &&
    (cd "$dir" && "./$name"); then
    PASSED+=("$name")
  else
    FAILED+=("$name")
  fi
}

build_and_run BoxModel "" "" \
  "$EXAMPLES/BoxModel/all_globals_mod.f" \
  "$EXAMPLES/BoxModel/head.f" \
  "$SCRIPT_DIR/gradient_check.f90" \
  "$SCRIPT_DIR/BoxModel/enzyme_driver.f90"

build_and_run FDC "" "" \
  "$EXAMPLES/FDC/head.f" \
  "$EXAMPLES/FDC/init.f" \
  "$SCRIPT_DIR/gradient_check.f90" \
  "$SCRIPT_DIR/FDC/enzyme_driver.f90"

SW=$EXAMPLES/ShallowWater
SW_SOURCES=()
for module in adsize size parms vars pfields force data weights mini; do
  SW_SOURCES+=("$SW/$module.f90")
done
for file in inifields readparms iowrapper read_data make_weights \
  map_control_vector small_routines_split small_routines_joint \
  cost_function cost_depth loop_body_wrapper_inner \
  loop_body_wrapper_outer forward_model ionetcdf; do
  SW_SOURCES+=("$SW/$file.F")
done
mkdir -p "$WORK/ShallowWater"
cp "$SW"/Data/* "$WORK/ShallowWater/"
grep '^grad(' "$SW/shallow_water_ad.out" | sed 's/grad(\([0-9 ]*\))=/\1/' \
  > "$WORK/ShallowWater/grad_reference.txt"
NETCDF_INCLUDE=${NETCDF_INCLUDE:-$(nc-config --includedir)}
build_and_run ShallowWater \
  "-fdefault-real-8 -fdefault-double-8 -I$SW/Inc -I$NETCDF_INCLUDE" \
  "-lnetcdff -lnetcdf" \
  "${SW_SOURCES[@]}" \
  "$SCRIPT_DIR/gradient_check.f90" \
  "$SCRIPT_DIR/ShallowWater/enzyme_driver.f90"

status=0
for name in "${PASSED[@]}"; do
  if [[ $EXPECTED_FAILURES == *" $name "* ]]; then
    echo "XPASS: $name (remove it from EXPECTED_FAILURES)"
    status=1
  else
    echo "PASS: $name"
  fi
done
for name in "${FAILED[@]}"; do
  if [[ $EXPECTED_FAILURES == *" $name "* ]]; then
    echo "XFAIL: $name"
  else
    echo "FAIL: $name"
    status=1
  fi
done
exit $status
