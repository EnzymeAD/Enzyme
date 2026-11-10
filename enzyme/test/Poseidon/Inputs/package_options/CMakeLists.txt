# Prints the compile options of each action target find_package(Poseidon)
# defines, of Poseidon::EnzymeOptions and of POSEIDON_ENZYME_OPTIONS, for
# package_gpu_runtime.cpp.
cmake_minimum_required(VERSION 3.18)
project(PoseidonPackageOptions NONE)

find_package(Poseidon CONFIG REQUIRED)
foreach(action Plain ProfileGenerate ProfileUse EnzymeOptions)
  get_target_property(options Poseidon::${action} INTERFACE_COMPILE_OPTIONS)
  message(STATUS "${POSEIDON_GPU_RUNTIME} ${action}: ${options}")
endforeach()
message(STATUS "${POSEIDON_GPU_RUNTIME} POSEIDON_ENZYME_OPTIONS: "
               "[${POSEIDON_ENZYME_OPTIONS}]")
