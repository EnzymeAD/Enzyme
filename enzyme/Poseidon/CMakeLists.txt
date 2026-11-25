# Poseidon: profile-guided numerical rewriting, built as its own pass plugin
# next to Enzyme's. Profile generation loads both (Poseidon first); a
# profile-use compile loads Poseidon alone.
if(NOT ENZYME_ENABLE_PLUGINS)
  message(FATAL_ERROR "ENZYME_POSEIDON needs ENZYME_ENABLE_PLUGINS")
endif()
if(LLVM_VERSION_MAJOR LESS 21)
  message(FATAL_ERROR "ENZYME_POSEIDON needs LLVM 21 or newer")
endif()
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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Build-tree layout mirrors an installation (bin/, lib/), which is what the
# driver resolves its plugin and runtimes against.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/bin)

find_package(Python3 COMPONENTS Interpreter QUIET)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
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
  # is checked.
  if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    if(NOT CMAKE_CUDA_ARCHITECTURES_NATIVE MATCHES
       "^[0-9]+-real(;[0-9]+-real)*$")
      message(
        FATAL_ERROR
          "CMAKE_CUDA_ARCHITECTURES is 'native', but CMake found no GPU; give "
          "the architecture of the GPU the runtimes are for, such as "
          "-DCMAKE_CUDA_ARCHITECTURES=120-real")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
  endif()
  # The architecture the integration tests compile for.
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "^([0-9]+)")
    set(POSEIDON_CUDA_ARCH sm_${CMAKE_MATCH_1} CACHE INTERNAL "")
  endif()
endif()
find_package(CUDAToolkit QUIET)

include(Herbie)

set(POSEIDON_SRC
    Core/CacheKey.cpp
    Sites/Canonicalize.cpp
    Model/CostModel.cpp
    Targets/MathLibrary.cpp
    Targets/Target.cpp
    Targets/Host.cpp
    Targets/NVPTX.cpp
    Targets/AMDGPU.cpp
    Targets/OffloadRuntime.cpp
    Pass/Driver.cpp
    Sites/Markers.cpp
    Sites/Annotations.cpp
    Sites/CallSites.cpp
    Profile/SiteProfile.cpp
    Pass/Twin.cpp
    Pass/Pipeline.cpp
    Model/Evaluators.cpp
    Techniques/Expansion.cpp
    Core/ExpansionModel.cpp
    Solve/Explain.cpp
    Pass/Flags.cpp
    Core/FPKind.cpp
    Techniques/Herbie.cpp
    Model/FPCore.cpp
    Core/HerbieRun.cpp
    Core/Options.cpp
    Core/Parallel.cpp
    Core/Solver.cpp
    Core/UniformReal.cpp
    Techniques/Matmul/HostDispatch.cpp
    Techniques/Matmul/InKernelRaise.cpp
    Profile/Instrument.cpp
    Sites/LaunchDescriptors.cpp
    Sites/LaunchStubs.cpp
    Solve/Optimize.cpp
    Sites/SiteIdentity.cpp
    Techniques/Precision.cpp
    Model/FPFormat.cpp
    Profile/ProfileRead.cpp
    Model/Sampling.cpp
    Solve/Solvers.cpp
    Solve/Pins.cpp
    Solve/Subgraphs.cpp
    Techniques/Staging.cpp
    Techniques/Unit.cpp
    Model/FPNode.cpp
    IR/Utils.cpp
    Techniques/Matmul/MatmulAccuracy.cpp
    Techniques/Matmul/MatmulCandidates.cpp
    Techniques/Matmul/MatmulSimulation.cpp
    Techniques/Matmul/RecognizeHostGemm.cpp
    Techniques/Matmul/RecognizeScalarLoop.cpp)

# The driver runs the clang of LLVM_DIR and parses command lines with its
# driver library, and the device profilers and HIP tools are compiled by it.
if(NOT Clang_FOUND OR NOT EXISTS ${LLVM_TOOLS_BINARY_DIR}/clang++)
  message(
    FATAL_ERROR
      "ENZYME_POSEIDON needs the clang of LLVM_DIR (${LLVM_TOOLS_BINARY_DIR}/clang++ and its CMake package); build LLVM with clang")
endif()

# The tools link LLVM and clang's driver library as LLVM's own tools do
# (add_llvm_executable, clang_target_link_libraries): the shared libraries
# where LLVM and clang were built to link them.
include(${CLANG_CMAKE_DIR}/AddClang.cmake)

# The cost table, which the calibrator reads as the plugin does.
add_library(poseidon_costtable OBJECT Core/CostTable.cpp)
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
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${ENZYME_SOURCE_DIR}/include
    ${ENZYME_SOURCE_DIR})
target_include_directories(
  ${POSEIDON_PLUGIN_TARGET}
  SYSTEM
  PRIVATE ${POSEIDON_MPFR_INCLUDE_DIR} ${POSEIDON_GMP_INCLUDE_DIR})
target_link_libraries(
  ${POSEIDON_PLUGIN_TARGET}
  PRIVATE ${MPFR_LIB_PATH} ${POSEIDON_GMP_LIB})

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

set_source_files_properties(
  Techniques/Herbie.cpp
  PROPERTIES
    COMPILE_DEFINITIONS
    POSEIDON_HERBIE_DEFAULT_BINARY="${POSEIDON_HERBIE_BINARY_PATH}")
# The accuracy model emulates the emitted FP32 expansions and the GEMM
# runtimes bit for bit, and the sampler reproduces the standard library's
# uniform reals bit for bit; each holds only if every operation is rounded as
# written.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set_source_files_properties(
    Core/ExpansionModel.cpp Core/UniformReal.cpp
    PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
  # The runtimes' directed roundings change the rounding mode.
  set_source_files_properties(
    Techniques/Matmul/MatmulSimulation.cpp
    PROPERTIES
      COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-frounding-math")
endif()
include(GNUInstallDirs)

# The device half of the GPU profiler as bitcode the plugin carries and links
# into each device module it compiles for profile generation:
# FPProfilerDevice<suffix>.bc compiled by COMPILE, embedded as the array
# kDeviceProfiler<suffix> that CONSUMER includes, compiled with DEFINES.
function(poseidon_embed_device_profiler suffix consumer)
  cmake_parse_arguments(PARSE_ARGV 2 _arg "" "" "COMPILE;DEFINES")
  set(_src ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler/FPProfilerDevice.cu)
  set(_bc ${CMAKE_CURRENT_BINARY_DIR}/FPProfilerDevice${suffix}.bc)
  set(_inc ${CMAKE_CURRENT_BINARY_DIR}/FPProfilerDevice${suffix}.inc)
  add_custom_command(
    OUTPUT ${_bc}
    COMMAND ${_arg_COMPILE} -emit-llvm -c ${_src} -o ${_bc}
    DEPENDS
      ${_src} ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler/FPProfilerCUDA.h
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/hip_as_cuda.h
    VERBATIM)
  add_custom_command(
    OUTPUT ${_inc}
    COMMAND
      ${CMAKE_COMMAND} -DIN=${_bc} -DOUT=${_inc} -DNAME=kDeviceProfiler${suffix}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake
    DEPENDS ${_bc} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake
    VERBATIM)
  string(TOLOWER "${suffix}" _target)
  if(_target)
    set(_target _${_target})
  endif()
  add_custom_target(poseidon_device_profiler${_target} DEPENDS ${_inc})
  add_dependencies(${POSEIDON_PLUGIN_TARGET} poseidon_device_profiler${_target})
  target_include_directories(
    ${POSEIDON_PLUGIN_TARGET}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  set_property(
    SOURCE ${consumer}
    APPEND
    PROPERTY COMPILE_DEFINITIONS ${_arg_DEFINES})
  set_property(SOURCE ${consumer} APPEND PROPERTY OBJECT_DEPENDS ${_inc})
endfunction()

# The architecture only admits the atomics the source uses; the code is
# generated for the GPU of the module the bitcode is linked into.
if(CMAKE_CUDA_COMPILER)
  get_filename_component(POSEIDON_CUDA_ROOT ${CUDAToolkit_BIN_DIR} DIRECTORY)
  poseidon_embed_device_profiler(
    ""
    Targets/NVPTX.cpp
    COMPILE
    ${LLVM_TOOLS_BINARY_DIR}/clang++
    -x
    cuda
    --cuda-device-only
    --cuda-gpu-arch=sm_60
    --cuda-path=${POSEIDON_CUDA_ROOT}
    -std=c++17
    -O3
    DEFINES
    POSEIDON_DEVICE_PROFILER)
endif()

# The ROCm installation HIP compiles take their device libraries and headers
# from, and the one AMD GPU processor the HIP runtimes, calibrators and device
# profiler are compiled for.
set(POSEIDON_ROCM_PATH ""
    CACHE PATH "ROCm installation HIP compiles use (--rocm-path)")
set(POSEIDON_AMDGPU_ARCH gfx942
    CACHE STRING "AMD GPU processor of the HIP builds (--offload-arch)")
if(NOT POSEIDON_AMDGPU_ARCH MATCHES "^gfx[0-9a-f]+$")
  message(
    FATAL_ERROR
      "POSEIDON_AMDGPU_ARCH='${POSEIDON_AMDGPU_ARCH}' is not "
      "one AMD GPU processor (gfx942, gfx90a, ...)")
endif()

add_library(poseidon_profile STATIC runtime/fpprofiler/FPProfiler.cpp)
target_include_directories(
  poseidon_profile
  PRIVATE ${ENZYME_SOURCE_DIR}/include)
# Linked into profile-generate builds only: it replaces malloc for the process.
add_library(poseidon_profile_host STATIC runtime/fpprofiler/FPProfilerHost.cpp)

# The error-corrected GEMM dispatch is cuMpSGEMM's; a build without it neither
# offers that candidate nor has its runtime.
option(POSEIDON_TCEC_USE_CUMPSGEMM
       "Build the error-corrected GEMM dispatch, which runs on cuMpSGEMM" OFF)
set(POSEIDON_CUMPSGEMM_ROOT ""
    CACHE PATH "cuMpSGEMM checkout for POSEIDON_TCEC_USE_CUMPSGEMM")
if(POSEIDON_TCEC_USE_CUMPSGEMM)
  if(NOT CMAKE_CUDA_COMPILER OR NOT POSEIDON_CUMPSGEMM_ROOT)
    message(
      FATAL_ERROR
        "POSEIDON_TCEC_USE_CUMPSGEMM needs a CUDA compiler and "
        "POSEIDON_CUMPSGEMM_ROOT")
  endif()
  target_compile_definitions(
    ${POSEIDON_PLUGIN_TARGET}
    PRIVATE POSEIDON_TCEC_USE_CUMPSGEMM)
endif()

if(CMAKE_CUDA_COMPILER)
  add_library(
    poseidon_rt
    STATIC
      runtime/native/native_rt.cu runtime/ozaki/ozaki_rt.cu
      runtime/direct/direct_rt.cu)
  # Whole-program per translation unit: every kernel is launched from the host
  # entry point in its own file, so the archive carries complete fatbins and an
  # application built by any compiler can link it.
  set_target_properties(poseidon_rt PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  # Optimized in every build type but Debug, as LLVM's own runtimes are;
  # -Xcompiler, since nvcc warns when its own -O is given twice.
  target_compile_options(
    poseidon_rt
    PRIVATE $<$<NOT:$<CONFIG:Debug>>:-Xcompiler=-O3>)
  # What a program that links poseidon_rt links besides it is said here only
  # (poseidon_rt_link_files).
  target_link_libraries(poseidon_rt PUBLIC CUDA::cublas CUDA::cudart)
  if(POSEIDON_TCEC_USE_CUMPSGEMM)
    target_sources(poseidon_rt PRIVATE runtime/tcec/tcec_rt.cu)
    target_include_directories(
      poseidon_rt
      PRIVATE ${POSEIDON_CUMPSGEMM_ROOT}/include)
    target_link_libraries(
      poseidon_rt
      PUBLIC ${POSEIDON_CUMPSGEMM_ROOT}/build/libcumpsgemm.so)
  endif()

  # The host half of the GPU profiler, linked by every profile-generate build
  # of CUDA code. It links no CUDA runtime: it takes the program's own, the
  # one the program's device modules are registered with.
  add_library(poseidon_profile_cuda STATIC runtime/fpprofiler/FPProfilerCUDA.cu)
  set_target_properties(
    poseidon_profile_cuda
    PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_include_directories(
    poseidon_profile_cuda
    PRIVATE ${ENZYME_SOURCE_DIR}/include)

  add_executable(poseidon-microbm tools/calibrate/microbm.cu)
  set(POSEIDON_TOOL_TARGETS poseidon-microbm)
  # The dispatch calibrator times the runtime entry points, so it links them.
  add_executable(poseidon-dispatch-calibrate
                 tools/calibrate/dispatch_calibrate.cu)
  target_compile_definitions(
    poseidon-dispatch-calibrate
    PRIVATE POSEIDON_TCEC=$<BOOL:${POSEIDON_TCEC_USE_CUMPSGEMM}>)
  target_include_directories(
    poseidon-dispatch-calibrate
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(
    poseidon-dispatch-calibrate
    PRIVATE poseidon_rt CUDA::cublas)
  list(APPEND POSEIDON_TOOL_TARGETS poseidon-dispatch-calibrate)
  # Installed, it still finds cuMpSGEMM where it was built, as the programs
  # poseidon-clang links do.
  if(POSEIDON_TCEC_USE_CUMPSGEMM)
    set_target_properties(
      poseidon-dispatch-calibrate
      PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
  endif()
endif()

# The device profiler, runtimes and calibrators compiled as HIP for
# POSEIDON_AMDGPU_ARCH by the clang the plugin is built against, each source
# whole as the CUDA ones are. TCEC has no HIP build.
set(POSEIDON_HIP_RUNTIMES OFF)
if(POSEIDON_ROCM_PATH AND NOT "AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  message(
    FATAL_ERROR
      "POSEIDON_ROCM_PATH needs an LLVM with the AMDGPU target; ${LLVM_DIR} has ${LLVM_TARGETS_TO_BUILD}")
endif()
if(POSEIDON_ROCM_PATH)
  set(POSEIDON_HIP_RUNTIMES ON)
  # hip_as_cuda.h, which every HIP build includes, includes hipBLAS.
  if(NOT EXISTS ${POSEIDON_ROCM_PATH}/lib/libhipblas.so)
    message(
      FATAL_ERROR
        "POSEIDON_ROCM_PATH=${POSEIDON_ROCM_PATH} has no hipBLAS "
        "(lib/libhipblas.so), which the HIP runtimes need")
  endif()
  # HIP marks its error type nodiscard; these sources drop the same results
  # their CUDA builds drop.
  set(_poseidon_hip_cxx
      ${LLVM_TOOLS_BINARY_DIR}/clang++
      -x
      hip
      --offload-arch=${POSEIDON_AMDGPU_ARCH}
      --rocm-path=${POSEIDON_ROCM_PATH}
      -std=c++17
      -O3
      -Wno-unused-value)

  # It links the ROCm device libraries of that processor (printf's hostcall),
  # which differ per ISA, so the plugin links it only into modules for that
  # processor.
  poseidon_embed_device_profiler(
    AMDGPU
    Targets/AMDGPU.cpp
    COMPILE
    ${_poseidon_hip_cxx}
    --cuda-device-only
    --no-gpu-bundle-output
    -fgpu-rdc
    DEFINES
    POSEIDON_DEVICE_PROFILER_AMDGPU
    POSEIDON_DEVICE_PROFILER_AMDGPU_ARCH="${POSEIDON_AMDGPU_ARCH}")

  set(POSEIDON_HIP_RUNTIME_HEADERS
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/hip_as_cuda.h
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/GemmDispatch.h
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/GemmRuntime.h
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler/FPProfilerCUDA.h
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler/FPProfileName.h
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler/ShadowTable.h
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ozaki/OzakiII.h
      ${ENZYME_SOURCE_DIR}/include/poseidon/poseidon.h)
  # The HIP headers define CUDA's directed-rounding intrinsics (the Ozaki-II
  # scaling's __dadd_ru, __fsub_rd, ...) only under
  # OCML_BASIC_ROUNDED_OPERATIONS.
  function(poseidon_add_hip_library name)
    set(_objs)
    foreach(_src ${ARGN})
      get_filename_component(_stem ${_src} NAME_WE)
      set(_obj ${CMAKE_CURRENT_BINARY_DIR}/hip/${name}/${_stem}.o)
      add_custom_command(
        OUTPUT ${_obj}
        COMMAND
          ${_poseidon_hip_cxx}
          -fPIC
          -DOCML_BASIC_ROUNDED_OPERATIONS
          -I${ENZYME_SOURCE_DIR}/include
          -c
          ${CMAKE_CURRENT_SOURCE_DIR}/${_src}
          -o
          ${_obj}
        DEPENDS
          ${CMAKE_CURRENT_SOURCE_DIR}/${_src} ${POSEIDON_HIP_RUNTIME_HEADERS}
        VERBATIM)
      list(APPEND _objs ${_obj})
    endforeach()
    add_library(${name} STATIC ${_objs})
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)
  endfunction()

  # Like poseidon_profile_cuda, it takes the program's own HIP runtime.
  poseidon_add_hip_library(poseidon_profile_hip
                           runtime/fpprofiler/FPProfilerCUDA.cu)
  poseidon_add_hip_library(poseidon_rt_hip runtime/native/native_rt.cu
                           runtime/ozaki/ozaki_rt.cu runtime/direct/direct_rt.cu)
  # In this build tree, the libraries of POSEIDON_ROCM_PATH; installed, those
  # of the ROCm installation of the machine the package is used on, which the
  # package configuration finds.
  target_link_libraries(
    poseidon_rt_hip
    INTERFACE
      $<BUILD_INTERFACE:${POSEIDON_ROCM_PATH}/lib/libhipblas.so>
      $<BUILD_INTERFACE:${POSEIDON_ROCM_PATH}/lib/libamdhip64.so>
      $<INSTALL_INTERFACE:Poseidon::rocm_hipblas>
      $<INSTALL_INTERFACE:Poseidon::rocm_amdhip64>)

  # The HIP runtime's own dependencies (the HSA runtime) resolve when a
  # calibrator runs, so a host without the GPU stack links them too.
  function(poseidon_add_hip_tool name src)
    set(_exe ${CMAKE_CURRENT_BINARY_DIR}/bin/${name})
    set(_libs)
    if(ARGN)
      set(_libs -x none ${ARGN})
    endif()
    add_custom_command(
      OUTPUT ${_exe}
      COMMAND
        ${_poseidon_hip_cxx}
        -I${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/${src}
        ${_libs}
        -L${POSEIDON_ROCM_PATH}/lib
        -lamdhip64
        -Wl,-rpath,${POSEIDON_ROCM_PATH}/lib
        -Wl,--allow-shlib-undefined
        -o
        ${_exe}
      DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/${src} ${POSEIDON_HIP_RUNTIME_HEADERS}
        ${ARGN}
      VERBATIM)
    add_custom_target(${name} ALL DEPENDS ${_exe})
  endfunction()

  poseidon_add_hip_tool(poseidon-microbm-hip tools/calibrate/microbm.cu)
  poseidon_add_hip_tool(
    poseidon-dispatch-calibrate-hip tools/calibrate/dispatch_calibrate.cu
    $<TARGET_FILE:poseidon_rt_hip> ${POSEIDON_ROCM_PATH}/lib/libhipblas.so)
  add_dependencies(poseidon-dispatch-calibrate-hip poseidon_rt_hip)
  set(POSEIDON_HIP_TOOL_TARGETS poseidon-microbm-hip
      poseidon-dispatch-calibrate-hip)
endif()

set(LLVM_LINK_COMPONENTS Support TargetParser)
add_llvm_executable(poseidon-calibrate tools/calibrate/poseidon-calibrate.cpp
                    $<TARGET_OBJECTS:poseidon_costtable>)
target_include_directories(
  poseidon-calibrate
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(
  poseidon-calibrate
  PRIVATE
    POSEIDON_CALIBRATE_TCEC=$<BOOL:${POSEIDON_TCEC_USE_CUMPSGEMM}>
    POSEIDON_CALIBRATE_HERBIE="${POSEIDON_HERBIE_BINARY_PATH}"
    POSEIDON_CALIBRATE_PYTHON="${Python3_EXECUTABLE}"
    POSEIDON_CALIBRATE_SRCDIR="${CMAKE_CURRENT_SOURCE_DIR}")
if(CMAKE_CUDA_COMPILER)
  add_dependencies(poseidon-calibrate poseidon-clang ${POSEIDON_TOOL_TARGETS})
endif()
if(POSEIDON_HIP_RUNTIMES)
  add_dependencies(poseidon-calibrate ${POSEIDON_HIP_TOOL_TARGETS})
endif()

add_executable(poseidon-probe tools/probe/poseidon-probe.cpp)

# Every HIP device compile is an LTO pre-link compile, in which Enzyme leaves a
# derivative request to the post-link run unless told otherwise, and the device
# link loads no plugin. poseidon-clang, POSEIDON_ENZYME_OPTIONS and
# Poseidon::EnzymeOptions give it to the HIP compiles that load Enzyme.
set(POSEIDON_ENZYME_HIP_OPTION "-enzyme-lto-prelink=1")

# The Enzyme plugin of this build tree, which differentiates the profiling
# probes, for the driver, the package and the tests.
set(POSEIDON_ENZYME_PLUGIN_NAME
    ClangEnzyme-${LLVM_VERSION_MAJOR}${CMAKE_SHARED_MODULE_SUFFIX})
set(POSEIDON_ENZYME_PLUGIN ""
    CACHE INTERNAL "The Enzyme clang plugin of this build tree")
if(TARGET ClangEnzyme-${LLVM_VERSION_MAJOR})
  set(POSEIDON_ENZYME_PLUGIN
      ${ENZYME_BINARY_DIR}/Enzyme/${POSEIDON_ENZYME_PLUGIN_NAME}
      CACHE INTERNAL "The Enzyme clang plugin of this build tree")
endif()

# The libraries poseidon_rt links, in its order: cuMpSGEMM defines cuBLAS's
# GEMM entry points and comes after cuBLAS, so that a call of cuBLAS reaches
# cuBLAS. Without TOOLKIT only those that are files, which a link adds beyond
# the CUDA toolkit's libraries; with it every one, as a file.
function(poseidon_rt_link_files out)
  cmake_parse_arguments(PARSE_ARGV 1 _arg "TOOLKIT" "" "")
  set(_files "")
  if(TARGET poseidon_rt)
    get_target_property(_libs poseidon_rt INTERFACE_LINK_LIBRARIES)
    foreach(_lib ${_libs})
      if(NOT TARGET ${_lib})
        list(APPEND _files ${_lib})
      elseif(_arg_TOOLKIT)
        get_target_property(_file ${_lib} LOCATION)
        list(APPEND _files ${_file})
      endif()
    endforeach()
  endif()
  set(${out} ${_files} PARENT_SCOPE)
endfunction()
poseidon_rt_link_files(POSEIDON_DRIVER_RT_LIBS)
# Comma-separated: a compile definition is no CMake list.
string(JOIN "," POSEIDON_DRIVER_RT_LIBS ${POSEIDON_DRIVER_RT_LIBS})

# One binary, two names: the C or the C++ clang, chosen from argv[0].
set(LLVM_LINK_COMPONENTS Option Support TargetParser)
add_llvm_executable(poseidon-clang tools/driver/poseidon-clang.cpp)
target_include_directories(poseidon-clang PRIVATE ${ENZYME_SOURCE_DIR}/include)
target_compile_definitions(
  poseidon-clang
  PRIVATE
    POSEIDON_DRIVER_CLANG="${LLVM_TOOLS_BINARY_DIR}/clang"
    POSEIDON_DRIVER_CLANGXX="${LLVM_TOOLS_BINARY_DIR}/clang++"
    POSEIDON_DRIVER_PLUGIN_NAME="${POSEIDON_PLUGIN_TARGET}${CMAKE_SHARED_MODULE_SUFFIX}"
    POSEIDON_DRIVER_PLUGIN="$<TARGET_FILE:${POSEIDON_PLUGIN_TARGET}>"
    POSEIDON_DRIVER_ENZYME="${POSEIDON_ENZYME_PLUGIN}"
    POSEIDON_DRIVER_ENZYME_HIP_OPTION="${POSEIDON_ENZYME_HIP_OPTION}"
    POSEIDON_DRIVER_INCLUDE="${ENZYME_SOURCE_DIR}/include"
    POSEIDON_DRIVER_LIBDIR="${CMAKE_CURRENT_BINARY_DIR}/lib"
    POSEIDON_DRIVER_BINDIR="$<TARGET_FILE_DIR:poseidon-clang>"
    POSEIDON_DRIVER_INSTALL_LIBDIR="${CMAKE_INSTALL_LIBDIR}"
    POSEIDON_DRIVER_INSTALL_INCLUDEDIR="${CMAKE_INSTALL_INCLUDEDIR}"
    POSEIDON_DRIVER_PROFILER_CUDA=$<BOOL:${CMAKE_CUDA_COMPILER}>
    POSEIDON_DRIVER_PROFILER_HIP=$<BOOL:${POSEIDON_HIP_RUNTIMES}>
    POSEIDON_DRIVER_RT_LIBS="${POSEIDON_DRIVER_RT_LIBS}"
    POSEIDON_DRIVER_CUDA_ROOT="${CUDAToolkit_TARGET_DIR}"
    POSEIDON_DRIVER_ROCM_ROOT="${POSEIDON_ROCM_PATH}")
clang_target_link_libraries(poseidon-clang PRIVATE clangDriver)
# Enzyme takes zlib out of LLVMSupport's interface for its table generator;
# clang's driver library compresses offload bundles with it.
if(LLVM_ENABLE_ZLIB AND NOT LLVM_LINK_LLVM_DYLIB)
  target_link_libraries(poseidon-clang PRIVATE ZLIB::ZLIB)
endif()
add_dependencies(poseidon-clang ${POSEIDON_PLUGIN_TARGET})
if(TARGET ClangEnzyme-${LLVM_VERSION_MAJOR})
  add_dependencies(poseidon-clang ClangEnzyme-${LLVM_VERSION_MAJOR})
endif()

add_custom_command(
  TARGET poseidon-clang
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink poseidon-clang
    $<TARGET_FILE_DIR:poseidon-clang>/poseidon-clang++)

install(
  TARGETS
  ${POSEIDON_PLUGIN_TARGET}
  poseidon_profile
  poseidon_profile_host
  poseidon-clang
  poseidon-calibrate
  poseidon-probe
  EXPORT PoseidonTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(
  CODE
  "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
             poseidon-clang
             \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/poseidon-clang++\")")
if(CMAKE_CUDA_COMPILER)
  install(
    TARGETS poseidon_rt poseidon_profile_cuda ${POSEIDON_TOOL_TARGETS}
    EXPORT PoseidonTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
if(POSEIDON_HIP_RUNTIMES)
  install(
    TARGETS poseidon_rt_hip poseidon_profile_hip
    EXPORT PoseidonTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  foreach(_tool ${POSEIDON_HIP_TOOL_TARGETS})
    install(
      PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/bin/${_tool}
      DESTINATION ${CMAKE_INSTALL_BINDIR})
  endforeach()
endif()
install(
  DIRECTORY ${ENZYME_SOURCE_DIR}/include/poseidon
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# The sources poseidon-calibrate runs.
install(
  FILES tools/calibrate/inkernel_calibrate.cu tools/calibrate/microbm_cpu.py
  tools/calibrate/herbie_platform.py
  DESTINATION ${CMAKE_INSTALL_DATADIR}/poseidon/tools/calibrate)
install(
  PROGRAMS tools/explain/poseidon-explain
  DESTINATION ${CMAKE_INSTALL_BINDIR})

# find_package(Poseidon CONFIG) from this build tree (Poseidon_DIR is this
# directory) and from an installation. The configuration lays out each tree's
# files and the Enzyme plugin of the same build, and defines the per-action
# interface targets (cmake/PoseidonConfig.cmake.in).
set(POSEIDON_VERSION 0.1)
set(POSEIDON_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/Poseidon)
set(POSEIDON_HAS_CUDA OFF)
if(CMAKE_CUDA_COMPILER)
  set(POSEIDON_HAS_CUDA ON)
endif()
set(POSEIDON_HAS_HIP ${POSEIDON_HIP_RUNTIMES})
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
set(POSEIDON_CONFIG_ENZYME_PLUGIN
    ${ENZYME_BINARY_DIR}/Enzyme/${POSEIDON_ENZYME_PLUGIN_NAME})
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
