# The runtime libraries a program Poseidon compiled links, in two families:
# the profiler (what a profile-generate build records) and the GEMM runtimes
# (the library calls a matrix product is rewritten into). A GPU library is
# built for each GPU runtime of the build (POSEIDON_HAS_CUDA, POSEIDON_HAS_HIP)
# from the same sources.

# The headers the runtime sources include, for the HIP compiles' dependencies.
set(POSEIDON_RUNTIME_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/DeviceApi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm/LibraryCalls.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm/LibraryCalls.inc
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm/GemmRuntime.h
    ${CMAKE_CURRENT_SOURCE_DIR}/profile/DeviceProfiler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/profile/ProfileFormat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/profile/ShadowTable.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm/SchemeConstants.h
    ${ENZYME_SOURCE_DIR}/include/poseidon/poseidon.h)
set(POSEIDON_RUNTIME_HEADERS ${POSEIDON_RUNTIME_HEADERS} PARENT_SCOPE)

# poseidon_add_gpu_runtime(<cuda target> <hip target> SOURCES <source>...)
# The static library of SOURCES for each GPU runtime of the build: <cuda
# target> compiled as CUDA, <hip target> as HIP by the plugin's clang
# (POSEIDON_HIP_CXX). Each source is compiled whole: every kernel is launched
# from a host function of its own file, so the archive carries complete device
# code and a program built by any compiler can link it. The HIP objects are
# compiled in and into the Poseidon build directory, which the CUID clang gives
# their symbols is derived from. The HIP headers define
# CUDA's directed-rounding intrinsics (the Ozaki-II scaling's __dadd_ru,
# __fsub_rd, ...) only under OCML_BASIC_ROUNDED_OPERATIONS.
function(poseidon_add_gpu_runtime cuda_target hip_target)
  cmake_parse_arguments(PARSE_ARGV 2 _arg "" "" "SOURCES")
  if(POSEIDON_HAS_CUDA)
    add_library(${cuda_target} STATIC ${_arg_SOURCES})
    set_target_properties(
      ${cuda_target}
      PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  endif()
  if(POSEIDON_HAS_HIP)
    set(_objs)
    foreach(_src ${_arg_SOURCES})
      get_filename_component(_stem ${_src} NAME_WE)
      set(_obj ${POSEIDON_BINARY_DIR}/hip/${hip_target}/${_stem}.o)
      add_custom_command(
        OUTPUT ${_obj}
        COMMAND
          ${POSEIDON_HIP_CXX}
          -fPIC
          -DOCML_BASIC_ROUNDED_OPERATIONS
          -I${ENZYME_SOURCE_DIR}/include
          -c
          ${CMAKE_CURRENT_SOURCE_DIR}/${_src}
          -o
          ${_obj}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_src} ${POSEIDON_RUNTIME_HEADERS}
        WORKING_DIRECTORY ${POSEIDON_BINARY_DIR}
        VERBATIM)
      list(APPEND _objs ${_obj})
    endforeach()
    add_library(${hip_target} STATIC ${_objs})
    set_target_properties(${hip_target} PROPERTIES LINKER_LANGUAGE CXX)
  endif()
endfunction()

# --- The profiler

# The host profiler, poseidon_metric, and the condition-number run: linked by
# every profile-generate build and, for poseidon_metric, by profile use.
add_library(poseidon_profile STATIC profile/Profiler.cpp)
target_include_directories(
  poseidon_profile
  PRIVATE ${ENZYME_SOURCE_DIR}/include)
# Linked into profile-generate builds only: it replaces malloc for the process.
add_library(poseidon_profile_shadows STATIC profile/Shadows.cpp)
install(
  TARGETS poseidon_profile poseidon_profile_shadows
  EXPORT PoseidonTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# The host half of the device profiler, linked by every profile-generate build
# of GPU code. It links no GPU runtime: it takes the program's own, the one the
# program's device modules are registered with.
poseidon_add_gpu_runtime(poseidon_profile_cuda poseidon_profile_hip SOURCES
                         profile/DeviceProfilerHost.cu)
if(POSEIDON_HAS_CUDA)
  target_include_directories(
    poseidon_profile_cuda
    PRIVATE ${ENZYME_SOURCE_DIR}/include)
endif()

# poseidon_embed_device_profiler(<suffix> <consumer> COMPILE <command>...
#                                DEFINES <definition>...)
# The device half of the profiler as bitcode the plugin carries and links into
# each device module it compiles for profile generation:
# DeviceProfilerDevice<suffix>.bc compiled by COMPILE, embedded as the array
# kDeviceProfiler<suffix> that the plugin source <consumer> includes, compiled
# with DEFINES.
function(poseidon_embed_device_profiler suffix consumer)
  cmake_parse_arguments(PARSE_ARGV 2 _arg "" "" "COMPILE;DEFINES")
  set(_src ${CMAKE_CURRENT_SOURCE_DIR}/profile/DeviceProfilerDevice.cu)
  set(_bc ${POSEIDON_BINARY_DIR}/DeviceProfilerDevice${suffix}.bc)
  set(_inc ${POSEIDON_BINARY_DIR}/DeviceProfilerDevice${suffix}.inc)
  add_custom_command(
    OUTPUT ${_bc}
    COMMAND ${_arg_COMPILE} -emit-llvm -c ${_src} -o ${_bc}
    DEPENDS
      ${_src} ${CMAKE_CURRENT_SOURCE_DIR}/profile/DeviceProfiler.h
      ${CMAKE_CURRENT_SOURCE_DIR}/DeviceApi.h
    WORKING_DIRECTORY ${POSEIDON_BINARY_DIR}
    VERBATIM)
  add_custom_command(
    OUTPUT ${_inc}
    COMMAND
      ${CMAKE_COMMAND} -DIN=${_bc} -DOUT=${_inc} -DNAME=kDeviceProfiler${suffix}
      -P ${POSEIDON_SOURCE_DIR}/cmake/EmbedFile.cmake
    DEPENDS ${_bc} ${POSEIDON_SOURCE_DIR}/cmake/EmbedFile.cmake
    VERBATIM)
  string(TOLOWER "${suffix}" _target)
  if(_target)
    set(_target _${_target})
  endif()
  add_custom_target(poseidon_device_profiler${_target} DEPENDS ${_inc})
  add_dependencies(${POSEIDON_PLUGIN_TARGET} poseidon_device_profiler${_target})
  target_include_directories(
    ${POSEIDON_PLUGIN_TARGET}
    PRIVATE ${POSEIDON_BINARY_DIR})
  set_property(
    SOURCE ${consumer}
    DIRECTORY ${POSEIDON_SOURCE_DIR}
    APPEND
    PROPERTY COMPILE_DEFINITIONS ${_arg_DEFINES})
  set_property(
    SOURCE ${consumer}
    DIRECTORY ${POSEIDON_SOURCE_DIR}
    APPEND
    PROPERTY OBJECT_DEPENDS ${_inc})
endfunction()

# The NVPTX architecture only admits the atomics the source uses; the code is
# generated for the GPU of the module the bitcode is linked into.
if(POSEIDON_HAS_CUDA)
  poseidon_embed_device_profiler(
    ""
    ${POSEIDON_SOURCE_DIR}/Targets/NVPTX.cpp
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
# It links the ROCm device libraries of that processor (printf's hostcall),
# which differ per ISA, so the plugin links it only into modules for that
# processor.
if(POSEIDON_HAS_HIP)
  poseidon_embed_device_profiler(
    AMDGPU
    ${POSEIDON_SOURCE_DIR}/Targets/AMDGPU.cpp
    COMPILE
    ${POSEIDON_HIP_CXX}
    --cuda-device-only
    --no-gpu-bundle-output
    -fgpu-rdc
    DEFINES
    POSEIDON_DEVICE_PROFILER_AMDGPU
    POSEIDON_DEVICE_PROFILER_AMDGPU_ARCH="${POSEIDON_AMDGPU_ARCH}")
endif()

# --- The GEMM runtimes

# Native, Ozaki-II and direct for every GPU runtime; TCEC on cuMpSGEMM for
# CUDA only.
poseidon_add_gpu_runtime(poseidon_gemm_cuda poseidon_gemm_hip SOURCES
                         gemm/OzakiII.cu gemm/Direct.cu)
if(POSEIDON_HAS_CUDA)
  # Optimized in every build type but Debug, as LLVM's own runtimes are;
  # -Xcompiler, since nvcc warns when its own -O is given twice.
  target_compile_options(
    poseidon_gemm_cuda
    PRIVATE $<$<NOT:$<CONFIG:Debug>>:-Xcompiler=-O3>)
  # What a program that links poseidon_gemm_cuda links besides it is said here
  # only
  # (poseidon_gemm_cuda_link_files).
  target_link_libraries(poseidon_gemm_cuda PUBLIC CUDA::cublas CUDA::cudart)
  if(POSEIDON_HAS_TCEC)
    target_sources(poseidon_gemm_cuda PRIVATE gemm/Tcec.cu)
    target_include_directories(
      poseidon_gemm_cuda
      PRIVATE ${POSEIDON_CUMPSGEMM_ROOT}/include)
    target_link_libraries(
      poseidon_gemm_cuda
      PUBLIC ${POSEIDON_CUMPSGEMM_ROOT}/build/libcumpsgemm.so)
  endif()
  install(
    TARGETS poseidon_gemm_cuda poseidon_profile_cuda
    EXPORT PoseidonTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(POSEIDON_HAS_HIP)
  # In this build tree, the libraries of POSEIDON_ROCM_PATH; installed, those
  # of the ROCm installation of the machine the package is used on, which the
  # package configuration finds.
  target_link_libraries(
    poseidon_gemm_hip
    INTERFACE
      $<BUILD_INTERFACE:${POSEIDON_ROCM_PATH}/lib/libhipblas.so>
      $<BUILD_INTERFACE:${POSEIDON_ROCM_PATH}/lib/libamdhip64.so>
      $<INSTALL_INTERFACE:Poseidon::rocm_hipblas>
      $<INSTALL_INTERFACE:Poseidon::rocm_amdhip64>)
  install(
    TARGETS poseidon_gemm_hip poseidon_profile_hip
    EXPORT PoseidonTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# poseidon_gemm_cuda_link_files(<out> [TOOLKIT])
# The libraries poseidon_gemm_cuda links, in its order: cuMpSGEMM defines
# cuBLAS's
# GEMM entry points and comes after cuBLAS, so that a call of cuBLAS reaches
# cuBLAS. Without TOOLKIT only those that are files, which a link adds beyond
# the CUDA toolkit's libraries; with it every one, as a file.
function(poseidon_gemm_cuda_link_files out)
  cmake_parse_arguments(PARSE_ARGV 1 _arg "TOOLKIT" "" "")
  set(_files "")
  if(TARGET poseidon_gemm_cuda)
    get_target_property(_libs poseidon_gemm_cuda INTERFACE_LINK_LIBRARIES)
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
