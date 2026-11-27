# Poseidon: profile-guided numerical rewriting, built as its own pass plugin
# next to Enzyme's. Profile generation loads both (Poseidon first); a
# profile-use compile loads Poseidon alone.
#
# This file declares what the build takes and builds the plugin;
# runtime/CMakeLists.txt builds the libraries a compiled program links and
# tools/CMakeLists.txt the programs a user runs. Each installs what it builds;
# the CMake package of both trees is configured here.

if(NOT ENZYME_ENABLE_PLUGINS)
  message(FATAL_ERROR "ENZYME_POSEIDON needs ENZYME_ENABLE_PLUGINS")
endif()
if(LLVM_VERSION_MAJOR LESS 21)
  message(FATAL_ERROR "ENZYME_POSEIDON needs LLVM 21 or newer")
endif()
# The driver runs the clang of LLVM_DIR and parses command lines with its
# driver library, and the device profilers and HIP runtimes are compiled by it.
if(NOT Clang_FOUND OR NOT EXISTS ${LLVM_TOOLS_BINARY_DIR}/clang++)
  message(
    FATAL_ERROR
      "ENZYME_POSEIDON needs the clang of LLVM_DIR (${LLVM_TOOLS_BINARY_DIR}/clang++ and its CMake package); build LLVM with clang")
endif()

# Where the sources and the build are, for the subdirectories and the tests.
set(POSEIDON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    CACHE INTERNAL "Poseidon's source directory")
set(POSEIDON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
    CACHE INTERNAL "Poseidon's build directory")

# --- Options

# The Herbie the rewriting pass runs unless a compile names one: a prebuilt
# executable, one built from source (cmake/Herbie.cmake), or none.
set(POSEIDON_HERBIE_BINARY ""
    CACHE FILEPATH "Prebuilt Herbie executable the pass runs by default")
option(POSEIDON_BUILD_HERBIE
       "Build Herbie from source, which takes Rust and Racket" OFF)
# The CUDA runtimes are built when CMake finds a CUDA compiler
# (POSEIDON_ENABLE_CUDA=Auto), or must be (On), or are not (Off), for
# CMAKE_CUDA_ARCHITECTURES (default: the GPUs of this machine, else a fixed
# list, below).
set(POSEIDON_ENABLE_CUDA Auto
    CACHE STRING "Build the CUDA runtimes: On, Off or Auto (default)")
# The HIP runtimes are built when POSEIDON_ROCM_PATH names a ROCm
# installation, for the one processor POSEIDON_AMDGPU_ARCH.
set(POSEIDON_ROCM_PATH ""
    CACHE PATH "ROCm installation HIP compiles use (--rocm-path)")
set(POSEIDON_AMDGPU_ARCH gfx942
    CACHE STRING "AMD GPU processor of the HIP builds (--offload-arch)")
# The error-corrected GEMM runtime (TCEC) is cuMpSGEMM's, built when
# POSEIDON_CUMPSGEMM_ROOT names a built checkout of it; without, the pass does
# not offer that candidate.
set(POSEIDON_CUMPSGEMM_ROOT ""
    CACHE PATH "Built cuMpSGEMM checkout of the TCEC runtime")

# --- What the build has

# MPFR's library is the one Enzyme found (MPFR_LIB_PATH). Its header, and
# GMP's, which it includes, are looked for next to the libraries and then in
# CMAKE_PREFIX_PATH, CMAKE_INCLUDE_PATH and the system: an MPFR or a GMP
# outside /usr is often in its own prefix.
get_filename_component(_poseidon_mpfr_libdir "${MPFR_LIB_PATH}" DIRECTORY)
find_path(
  POSEIDON_MPFR_INCLUDE_DIR mpfr.h
  HINTS ${_poseidon_mpfr_libdir}/../include)
find_library(POSEIDON_GMP_LIB gmp HINTS ${_poseidon_mpfr_libdir})
get_filename_component(_poseidon_gmp_libdir "${POSEIDON_GMP_LIB}" DIRECTORY)
find_path(
  POSEIDON_GMP_INCLUDE_DIR gmp.h
  HINTS ${_poseidon_gmp_libdir}/../include ${POSEIDON_MPFR_INCLUDE_DIR})
set(_poseidon_missing)
foreach(_found MPFR_LIB_PATH POSEIDON_MPFR_INCLUDE_DIR POSEIDON_GMP_LIB
        POSEIDON_GMP_INCLUDE_DIR)
  if(NOT ${_found})
    list(APPEND _poseidon_missing ${_found})
  endif()
endforeach()
if(_poseidon_missing)
  string(REPLACE ";" ", " _poseidon_missing "${_poseidon_missing}")
  message(
    FATAL_ERROR
      "ENZYME_POSEIDON needs MPFR and GMP, libraries and headers; not found: "
      "${_poseidon_missing}. Add the installation prefix of each to "
      "CMAKE_PREFIX_PATH.")
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)

string(TOUPPER "${POSEIDON_ENABLE_CUDA}" _poseidon_enable_cuda)
if(NOT _poseidon_enable_cuda MATCHES "^(ON|OFF|AUTO)$")
  message(
    FATAL_ERROR
      "POSEIDON_ENABLE_CUDA='${POSEIDON_ENABLE_CUDA}' is not On, Off or Auto")
endif()
set(POSEIDON_HAS_CUDA OFF)
if(NOT _poseidon_enable_cuda STREQUAL "OFF")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(POSEIDON_HAS_CUDA ON)
  elseif(_poseidon_enable_cuda STREQUAL "ON")
    message(
      FATAL_ERROR
        "POSEIDON_ENABLE_CUDA is On, but CMake found no CUDA compiler; set "
        "CMAKE_CUDA_COMPILER to nvcc")
  endif()
endif()
# The architecture the integration tests compile for; none takes the tests'
# default.
set(POSEIDON_CUDA_ARCH "" CACHE INTERNAL "")
if(POSEIDON_HAS_CUDA)
  if(CMAKE_VERSION VERSION_LESS 3.24)
    message(FATAL_ERROR "Poseidon's CUDA runtimes need CMake 3.24 or newer")
  endif()
  if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
  endif()
  enable_language(CUDA)
  # nvcc resolves `native` at every compile, to a JIT-only architecture where
  # it sees no GPU; the GPUs CMake found here are used instead. A failed
  # detection can leave its message in place of the list, so the list's form
  # is checked. On a machine without a GPU the runtimes are built for every
  # architecture from Turing (sm_75, nvcc's oldest without a deprecation
  # warning) to Hopper, and as PTX that the driver compiles for newer ones.
  set(_poseidon_cuda_arch_source "")
  if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    if(CMAKE_CUDA_ARCHITECTURES_NATIVE MATCHES "^[0-9]+-real(;[0-9]+-real)*$")
      set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
    else()
      set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 90-real 90-virtual)
      set(_poseidon_cuda_arch_source
          " (CMake found no GPU; CMAKE_CUDA_ARCHITECTURES chooses others)")
    endif()
  endif()
  message(
    STATUS
      "Poseidon: CUDA runtimes for ${CMAKE_CUDA_ARCHITECTURES}${_poseidon_cuda_arch_source}")
  if(NOT _poseidon_cuda_arch_source
     AND CMAKE_CUDA_ARCHITECTURES MATCHES "^([0-9]+)")
    set(POSEIDON_CUDA_ARCH sm_${CMAKE_MATCH_1} CACHE INTERNAL "")
  endif()
elseif(_poseidon_enable_cuda STREQUAL "OFF")
  message(STATUS "Poseidon: no CUDA runtimes (POSEIDON_ENABLE_CUDA=Off)")
else()
  message(STATUS "Poseidon: no CUDA compiler found, so no CUDA runtimes")
endif()
# The toolkit the CUDA runtimes link and the driver and the device profiler
# compile against (its root, the directory above nvcc's).
find_package(CUDAToolkit QUIET)
get_filename_component(POSEIDON_CUDA_ROOT "${CUDAToolkit_BIN_DIR}" DIRECTORY)

if(POSEIDON_ROCM_PATH AND NOT "AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  message(
    FATAL_ERROR
      "POSEIDON_ROCM_PATH needs an LLVM with the AMDGPU target; ${LLVM_DIR} has ${LLVM_TARGETS_TO_BUILD}")
endif()
if(NOT POSEIDON_AMDGPU_ARCH MATCHES "^gfx[0-9a-f]+$")
  message(
    FATAL_ERROR
      "POSEIDON_AMDGPU_ARCH='${POSEIDON_AMDGPU_ARCH}' is not "
      "one AMD GPU processor (gfx942, gfx90a, ...)")
endif()
if(POSEIDON_CUMPSGEMM_ROOT AND NOT POSEIDON_HAS_CUDA)
  message(FATAL_ERROR "POSEIDON_CUMPSGEMM_ROOT needs the CUDA runtimes")
endif()

# Which GPU runtimes this build has, for the runtimes, the tools, the package
# and the tests.
set(POSEIDON_HAS_HIP OFF)
if(POSEIDON_ROCM_PATH)
  set(POSEIDON_HAS_HIP ON)
endif()
set(POSEIDON_HAS_TCEC OFF)
if(POSEIDON_CUMPSGEMM_ROOT)
  set(POSEIDON_HAS_TCEC ON)
endif()
foreach(_runtime CUDA HIP TCEC)
  set(POSEIDON_HAS_${_runtime} ${POSEIDON_HAS_${_runtime}}
      CACHE INTERNAL "Whether this build has the ${_runtime} runtime")
endforeach()

# How a HIP source is compiled: by the clang the plugin is built against, for
# POSEIDON_AMDGPU_ARCH. HIP marks its error type nodiscard; the sources drop
# the same results their CUDA builds drop.
if(POSEIDON_HAS_HIP)
  # DeviceApi.h, which every GPU runtime source includes, includes hipBLAS.
  if(NOT EXISTS ${POSEIDON_ROCM_PATH}/lib/libhipblas.so)
    message(
      FATAL_ERROR
        "POSEIDON_ROCM_PATH=${POSEIDON_ROCM_PATH} has no hipBLAS "
        "(lib/libhipblas.so), which the HIP runtimes need")
  endif()
  set(POSEIDON_HIP_CXX
      ${LLVM_TOOLS_BINARY_DIR}/clang++
      -x
      hip
      --offload-arch=${POSEIDON_AMDGPU_ARCH}
      --rocm-path=${POSEIDON_ROCM_PATH}
      -std=c++17
      -O3
      -Wno-unused-value)
endif()

# The Enzyme plugin of this build, which differentiates the profiling probes,
# for the driver, the package and the tests: POSEIDON_ENZYME_PLUGIN is empty
# in a build without it.
set(POSEIDON_ENZYME_PLUGIN_NAME
    ClangEnzyme-${LLVM_VERSION_MAJOR}${CMAKE_SHARED_MODULE_SUFFIX})
set(POSEIDON_ENZYME_PLUGIN_PATH
    ${ENZYME_BINARY_DIR}/Enzyme/${POSEIDON_ENZYME_PLUGIN_NAME})
set(_poseidon_enzyme_plugin "")
if(TARGET ClangEnzyme-${LLVM_VERSION_MAJOR})
  set(_poseidon_enzyme_plugin ${POSEIDON_ENZYME_PLUGIN_PATH})
endif()
set(POSEIDON_ENZYME_PLUGIN "${_poseidon_enzyme_plugin}"
    CACHE INTERNAL "The Enzyme clang plugin of this build tree")
# Every HIP device compile is an LTO pre-link compile, in which Enzyme leaves a
# derivative request to the post-link run unless told otherwise, and the device
# link loads no plugin. poseidon-clang, POSEIDON_ENZYME_OPTIONS and
# Poseidon::EnzymeOptions give it to the HIP compiles that load Enzyme.
set(POSEIDON_ENZYME_HIP_OPTION "-enzyme-lto-prelink=1")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)

# Build-tree layout mirrors an installation (bin/, lib/), which is what the
# driver resolves its plugin and runtimes against.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/bin)

# The tools link LLVM and clang's driver library as LLVM's own tools do
# (add_llvm_executable, clang_target_link_libraries): the shared libraries
# where LLVM and clang were built to link them.
include(${CLANG_CMAKE_DIR}/AddClang.cmake)

# --- The plugin

set(POSEIDON_SRC
    Support/CacheKey.cpp
    Sites/Canonicalize.cpp
    Model/CostModel.cpp
    Targets/MathLibrary.cpp
    Targets/Target.cpp
    Targets/NVPTX.cpp
    Targets/AMDGPU.cpp
    Targets/OffloadRuntime.cpp
    Pass/Actions.cpp
    Sites/Markers.cpp
    Sites/Annotations.cpp
    Sites/CallSites.cpp
    Profile/InstrumentSite.cpp
    Pass/AsWritten.cpp
    Pass/Pipeline.cpp
    Model/Evaluators.cpp
    Techniques/Precision/Expansion.cpp
    Techniques/Precision/ExpansionModel.cpp
    Solve/Explain.cpp
    Pass/Flags.cpp
    Techniques/Algebraic/Algebraic.cpp
    Core/FPCore.cpp
    Techniques/Algebraic/HerbieRun.cpp
    Support/Parallel.cpp
    Solve/DP.cpp
    Support/UniformReal.cpp
    Techniques/Matmul/RealizeLibraryCall.cpp
    Techniques/Matmul/RealizeInKernel.cpp
    Profile/Instrument.cpp
    Sites/LaunchManifest.cpp
    Solve/Optimize.cpp
    Sites/Site.cpp
    Sites/SiteIdentity.cpp
    Techniques/Precision/Precision.cpp
    Core/NumberFormat.cpp
    Profile/ProfileReader.cpp
    Model/Sampling.cpp
    Solve/SolveSites.cpp
    Solve/Pins.cpp
    Solve/Subgraphs.cpp
    Techniques/Precision/Narrowing.cpp
    Core/Unit.cpp
    Model/SubgraphUnit.cpp
    Techniques/Techniques.cpp
    Core/FPNode.cpp
    Support/Utils.cpp
    Support/Diagnostics.cpp
    Techniques/Matmul/Accuracy.cpp
    Techniques/Matmul/Matmul.cpp
    Techniques/Matmul/Simulation.cpp
    Techniques/Matmul/Recognize.cpp
    Techniques/Matmul/RecognizeRuntimeShape.cpp
    Techniques/Matmul/RecognizeConstantShape.cpp)

# The cost table, which the calibrator reads as the plugin does.
add_library(poseidon_costtable OBJECT Model/CostTable.cpp)
set_target_properties(
  poseidon_costtable
  PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  poseidon_costtable
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set(POSEIDON_PLUGIN_TARGET Poseidon-${LLVM_VERSION_MAJOR})
add_llvm_library(
  ${POSEIDON_PLUGIN_TARGET}
  ${POSEIDON_SRC}
  $<TARGET_OBJECTS:poseidon_costtable>
  MODULE
  PARTIAL_SOURCES_INTENDED
  PLUGIN_TOOL
  clang
  DEPENDS
  intrinsics_gen)
target_include_directories(
  ${POSEIDON_PLUGIN_TARGET}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${ENZYME_SOURCE_DIR}/include)
# Enzyme's helpers, of which the plugin calls only those that touch no Enzyme
# state (test/Poseidon/Pass/enzyme_symbols.ll). A system directory, so that
# the helpers a source file does not call do not warn as unused.
target_include_directories(
  ${POSEIDON_PLUGIN_TARGET}
  SYSTEM
  PRIVATE ${ENZYME_SOURCE_DIR})
target_include_directories(
  ${POSEIDON_PLUGIN_TARGET}
  SYSTEM
  PRIVATE ${POSEIDON_MPFR_INCLUDE_DIR} ${POSEIDON_GMP_INCLUDE_DIR})
target_link_libraries(
  ${POSEIDON_PLUGIN_TARGET}
  PRIVATE ${MPFR_LIB_PATH} ${POSEIDON_GMP_LIB})
if(POSEIDON_HAS_TCEC)
  target_compile_definitions(
    ${POSEIDON_PLUGIN_TARGET}
    PRIVATE POSEIDON_TCEC_USE_CUMPSGEMM)
endif()

# What a cached result was computed by: Poseidon's sources. Hashed when they
# change, not when the build is configured.
file(GLOB_RECURSE POSEIDON_SOURCE_FILES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
# A source removed or moved leaves no newer time stamp behind, so the list is a
# dependency too; GENERATE rewrites it only when it changes.
string(REPLACE ";" "\n" _poseidon_source_list "${POSEIDON_SOURCE_FILES}")
file(
  GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/identity/Sources.txt
  CONTENT "${_poseidon_source_list}\n")
set(POSEIDON_BUILD_IDENTITY ${CMAKE_CURRENT_BINARY_DIR}/identity/BuildIdentity.h)
add_custom_command(
  OUTPUT ${POSEIDON_BUILD_IDENTITY}
  COMMAND
    ${CMAKE_COMMAND} -DPASS_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DOUT=${POSEIDON_BUILD_IDENTITY} -P
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildIdentity.cmake
  DEPENDS
    ${POSEIDON_SOURCE_FILES} ${CMAKE_CURRENT_BINARY_DIR}/identity/Sources.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildIdentity.cmake
  VERBATIM)
add_custom_target(poseidon_build_identity DEPENDS ${POSEIDON_BUILD_IDENTITY})
add_dependencies(${POSEIDON_PLUGIN_TARGET} poseidon_build_identity)
target_include_directories(
  ${POSEIDON_PLUGIN_TARGET}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/identity)

# Every operation of the plugin is rounded as written, so that its accuracy
# estimates, and the decisions made from them, are the same on every host: a
# compiler fuses a multiply and an add where the host's base instruction set
# has a fused multiply-add (AArch64) and not where it has none (x86-64). The
# accuracy model also emulates the emitted FP32 expansions and the GEMM
# runtimes bit for bit, and the sampler reproduces the standard library's
# uniform reals bit for bit.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(
    ${POSEIDON_PLUGIN_TARGET}
    PRIVATE -fno-fast-math -ffp-contract=off)
  # The runtimes' directed roundings change the rounding mode.
  set_source_files_properties(
    Techniques/Matmul/Simulation.cpp
    PROPERTIES COMPILE_OPTIONS "-frounding-math")
endif()

install(
  TARGETS ${POSEIDON_PLUGIN_TARGET}
  EXPORT PoseidonTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(
  DIRECTORY ${ENZYME_SOURCE_DIR}/include/poseidon
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(runtime)
add_subdirectory(tools)

# The Herbie the pass runs by default (tools/CMakeLists.txt).
set_source_files_properties(
  Pass/Flags.cpp
  PROPERTIES
    COMPILE_DEFINITIONS
    POSEIDON_HERBIE_DEFAULT_BINARY="${POSEIDON_HERBIE_BINARY_PATH}")

# --- The CMake package

# find_package(Poseidon CONFIG) from this build tree (Poseidon_DIR is this
# directory) and from an installation. The configuration lays out each tree's
# files and the Enzyme plugin of the same build, and defines the per-action
# interface targets (cmake/PoseidonConfig.cmake.in).
set(POSEIDON_VERSION 0.1)
set(POSEIDON_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/Poseidon)
install(
  EXPORT PoseidonTargets
  NAMESPACE Poseidon::
  DESTINATION ${POSEIDON_INSTALL_CMAKEDIR})
export(
  EXPORT PoseidonTargets
  NAMESPACE Poseidon::
  FILE ${CMAKE_CURRENT_BINARY_DIR}/PoseidonTargets.cmake)

include(CMakePackageConfigHelpers)
set(POSEIDON_CONFIG_INCLUDE_DIR ${ENZYME_SOURCE_DIR}/include)
set(POSEIDON_CONFIG_ENZYME_PLUGIN ${POSEIDON_ENZYME_PLUGIN_PATH})
set(POSEIDON_CONFIG_FINDS_ROCM FALSE)
configure_package_config_file(
  cmake/PoseidonConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/PoseidonConfig.cmake
  INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  PATH_VARS POSEIDON_CONFIG_INCLUDE_DIR POSEIDON_CONFIG_ENZYME_PLUGIN)
# Enzyme installs its plugins to lib/.
set(POSEIDON_CONFIG_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(POSEIDON_CONFIG_ENZYME_PLUGIN lib/${POSEIDON_ENZYME_PLUGIN_NAME})
set(POSEIDON_CONFIG_FINDS_ROCM TRUE)
configure_package_config_file(
  cmake/PoseidonConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PoseidonConfig.cmake
  INSTALL_DESTINATION ${POSEIDON_INSTALL_CMAKEDIR}
  PATH_VARS POSEIDON_CONFIG_INCLUDE_DIR POSEIDON_CONFIG_ENZYME_PLUGIN)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/PoseidonConfigVersion.cmake
  VERSION ${POSEIDON_VERSION}
  COMPATIBILITY AnyNewerVersion)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PoseidonConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/PoseidonConfigVersion.cmake
  DESTINATION ${POSEIDON_INSTALL_CMAKEDIR})
