# A HIP project that uses Poseidon through find_package(Poseidon), for
# full_compile_amdgpu.hip: SOURCE compiled as HIP into the object libraries k
# and k_rdc (relocatable device code) under the action target ACTION names, the
# GPU language from an interface target linked before it. With LOAD_ENZYME the
# project loads Enzyme's plugin itself, after the action target, and with
# ENZYME_OPTIONS links Poseidon::EnzymeOptions to the targets that load it.
# Objects only; nothing links.
cmake_minimum_required(VERSION 3.18)
project(PoseidonPackageHIP LANGUAGES CXX)

set(SOURCE "" CACHE FILEPATH "HIP source")
set(AMDGPU_ARCH "" CACHE STRING "Processor the device code is compiled for")
set(ROCM_PATH "" CACHE PATH "ROCm installation")
set(ACTION ProfileGenerate CACHE STRING "Plain, ProfileGenerate or ProfileUse")
set(POSEIDON_FLAGS "" CACHE STRING "Additional -poseidon-<flag>s")
set(DEFINES "" CACHE STRING "Preprocessor definitions")
option(LOAD_ENZYME "Load Enzyme's plugin in the project's own options" OFF)
option(ENZYME_OPTIONS "Link Poseidon::EnzymeOptions where Enzyme is loaded"
       OFF)

set(POSEIDON_GPU_RUNTIME HIP)
find_package(Poseidon CONFIG REQUIRED)

add_library(hip_device INTERFACE)
target_compile_options(
  hip_device INTERFACE -x hip --offload-arch=${AMDGPU_ARCH}
                       --rocm-path=${ROCM_PATH})

add_library(project_enzyme INTERFACE)
target_compile_options(project_enzyme
                       INTERFACE -fpass-plugin=${POSEIDON_ENZYME_PLUGIN})
if(POSEIDON_LLVM_VERSION_MAJOR LESS 22)
  target_compile_options(
    project_enzyme
    INTERFACE "SHELL:-Xclang -load -Xclang ${POSEIDON_ENZYME_PLUGIN}")
endif()

separate_arguments(POSEIDON_FLAGS UNIX_COMMAND "${POSEIDON_FLAGS}")
list(TRANSFORM POSEIDON_FLAGS PREPEND "SHELL:${POSEIDON_MLLVM} ")

configure_file(${SOURCE} ${CMAKE_BINARY_DIR}/k.cpp COPYONLY)
foreach(target k k_rdc)
  add_library(${target} OBJECT ${CMAKE_BINARY_DIR}/k.cpp)
  target_link_libraries(${target} PRIVATE hip_device Poseidon::${ACTION})
  if(LOAD_ENZYME)
    target_link_libraries(${target} PRIVATE project_enzyme)
    if(ENZYME_OPTIONS)
      target_link_libraries(${target} PRIVATE Poseidon::EnzymeOptions)
    endif()
  endif()
  if(POSEIDON_FLAGS)
    target_compile_options(${target} PRIVATE ${POSEIDON_FLAGS})
  endif()
  if(DEFINES)
    target_compile_definitions(${target} PRIVATE ${DEFINES})
  endif()
endforeach()
target_compile_options(k_rdc PRIVATE -fgpu-rdc)
