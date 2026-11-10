# Poseidon: profile-guided numerical rewriting, built as its own pass plugin
# next to Enzyme's. Profile generation loads both (Poseidon first); a
# profile-use compile loads Poseidon alone.
if(NOT ENZYME_ENABLE_PLUGINS)
  message(FATAL_ERROR "ENZYME_POSEIDON needs ENZYME_ENABLE_PLUGINS")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Build-tree layout mirrors an installation (bin/, lib/), which is what the
# driver resolves its plugin and runtimes against.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/bin)

find_package(MPFR REQUIRED)
find_package(Python3 COMPONENTS Interpreter QUIET)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
  endif()
  enable_language(CUDA)
endif()
find_package(CUDAToolkit QUIET)

include(Herbie)

set(POSEIDON_SRC
    Plugin.cpp
    CacheKey.cpp
    Canonicalize.cpp
    CostModel.cpp
    DeviceMath.cpp
    DeviceProfile.cpp
    Driver.cpp
    Evaluators.cpp
    Expansion.cpp
    Explain.cpp
    Flags.cpp
    Herbie.cpp
    HerbieRun.cpp
    HostDispatch.cpp
    InKernelRaise.cpp
    Instrument.cpp
    LaunchDescriptors.cpp
    Optimize.cpp
    Precision.cpp
    ProfileRead.cpp
    RaiseWMMA.cpp
    Sampling.cpp
    Solvers.cpp
    Staging.cpp
    Types.cpp
    Utils.cpp
    WmmaUtils.cpp
    MatmulAccuracy.cpp
    MatmulApply.cpp
    MatmulCandidates.cpp
    MatmulPricing.cpp
    MatmulProfile.cpp
    RecognizeHostGemm.cpp)

set(POSEIDON_PLUGIN_TARGET Poseidon-${LLVM_VERSION_MAJOR})
if(Clang_FOUND)
  add_llvm_library(${POSEIDON_PLUGIN_TARGET} ${POSEIDON_SRC} MODULE
                   PARTIAL_SOURCES_INTENDED PLUGIN_TOOL clang)
else()
  add_llvm_library(${POSEIDON_PLUGIN_TARGET} ${POSEIDON_SRC} MODULE
                   PARTIAL_SOURCES_INTENDED PLUGIN_TOOL opt)
endif()
target_include_directories(
  ${POSEIDON_PLUGIN_TARGET}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${ENZYME_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler)
target_link_libraries(${POSEIDON_PLUGIN_TARGET} PRIVATE MPFR::MPFR)
set_source_files_properties(
  Herbie.cpp
  PROPERTIES COMPILE_DEFINITIONS HERBIE_BINARY="${POSEIDON_HERBIE_BINARY_PATH}")
include(GNUInstallDirs)
# Where a compile looks for the cost model of its target device when neither
# the flag nor the environment names one.
set_source_files_properties(
  CostModel.cpp
  PROPERTIES
    COMPILE_DEFINITIONS
    POSEIDON_INSTALL_COST_MODELS="${CMAKE_INSTALL_FULL_DATADIR}/poseidon/cost_models")

# The device half of the CUDA profiler, as NVPTX bitcode the plugin carries and
# links into each device module it compiles for profile generation. The
# architecture only admits the atomics the source uses; the code is generated
# for the GPU of the module it is linked into.
if(CMAKE_CUDA_COMPILER)
  get_filename_component(POSEIDON_CUDA_ROOT ${CUDAToolkit_BIN_DIR} DIRECTORY)
  set(POSEIDON_DEVICE_PROFILER_SRC
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler/FPProfilerDevice.cu)
  set(POSEIDON_DEVICE_PROFILER_BC ${CMAKE_CURRENT_BINARY_DIR}/FPProfilerDevice.bc)
  set(POSEIDON_DEVICE_PROFILER_INC
      ${CMAKE_CURRENT_BINARY_DIR}/FPProfilerDevice.inc)
  add_custom_command(
    OUTPUT ${POSEIDON_DEVICE_PROFILER_BC}
    COMMAND
      ${LLVM_TOOLS_BINARY_DIR}/clang++ -x cuda --cuda-device-only
      --cuda-gpu-arch=sm_60 --cuda-path=${POSEIDON_CUDA_ROOT} -std=c++17 -O3
      -emit-llvm -c -I${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler
      ${POSEIDON_DEVICE_PROFILER_SRC} -o ${POSEIDON_DEVICE_PROFILER_BC}
    DEPENDS ${POSEIDON_DEVICE_PROFILER_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler/FPProfilerCUDA.h
    VERBATIM)
  add_custom_command(
    OUTPUT ${POSEIDON_DEVICE_PROFILER_INC}
    COMMAND
      ${CMAKE_COMMAND} -DIN=${POSEIDON_DEVICE_PROFILER_BC}
      -DOUT=${POSEIDON_DEVICE_PROFILER_INC} -DNAME=kDeviceProfiler -P
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake
    DEPENDS ${POSEIDON_DEVICE_PROFILER_BC}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake
    VERBATIM)
  add_custom_target(poseidon_device_profiler
                    DEPENDS ${POSEIDON_DEVICE_PROFILER_INC})
  add_dependencies(${POSEIDON_PLUGIN_TARGET} poseidon_device_profiler)
  target_include_directories(${POSEIDON_PLUGIN_TARGET}
                             PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  set_source_files_properties(
    DeviceProfile.cpp
    PROPERTIES COMPILE_DEFINITIONS POSEIDON_DEVICE_PROFILER OBJECT_DEPENDS
               ${POSEIDON_DEVICE_PROFILER_INC})
endif()

# The ROCm installation HIP compiles take their device libraries and headers
# from, and the one AMD GPU processor the HIP runtimes, calibrators and device
# profiler are compiled for.
set(POSEIDON_ROCM_PATH ""
    CACHE PATH "ROCm installation HIP compiles use (--rocm-path)")
set(POSEIDON_AMDGPU_ARCH gfx942
    CACHE STRING "AMD GPU processor of the HIP builds (--offload-arch)")
if(NOT POSEIDON_AMDGPU_ARCH MATCHES "^gfx[0-9a-f]+$")
  message(FATAL_ERROR "POSEIDON_AMDGPU_ARCH='${POSEIDON_AMDGPU_ARCH}' is not "
                      "one AMD GPU processor (gfx942, gfx90a, ...)")
endif()

# The same device half as AMDGPU bitcode, for HIP device modules. It links the
# ROCm device libraries of that processor (printf's hostcall), which differ per
# ISA, so the plugin links it only into modules for that processor.
if(POSEIDON_ROCM_PATH AND "AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(POSEIDON_DEVICE_PROFILER_AMDGPU_BC
      ${CMAKE_CURRENT_BINARY_DIR}/FPProfilerDeviceAMDGPU.bc)
  set(POSEIDON_DEVICE_PROFILER_AMDGPU_INC
      ${CMAKE_CURRENT_BINARY_DIR}/FPProfilerDeviceAMDGPU.inc)
  add_custom_command(
    OUTPUT ${POSEIDON_DEVICE_PROFILER_AMDGPU_BC}
    COMMAND
      ${LLVM_TOOLS_BINARY_DIR}/clang++ -x hip --cuda-device-only
      --no-gpu-bundle-output -fgpu-rdc --offload-arch=${POSEIDON_AMDGPU_ARCH}
      --rocm-path=${POSEIDON_ROCM_PATH} -std=c++17 -O3 -emit-llvm -c
      -I${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler/FPProfilerDevice.cu -o
      ${POSEIDON_DEVICE_PROFILER_AMDGPU_BC}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler/FPProfilerDevice.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler/FPProfilerCUDA.h
    VERBATIM)
  add_custom_command(
    OUTPUT ${POSEIDON_DEVICE_PROFILER_AMDGPU_INC}
    COMMAND
      ${CMAKE_COMMAND} -DIN=${POSEIDON_DEVICE_PROFILER_AMDGPU_BC}
      -DOUT=${POSEIDON_DEVICE_PROFILER_AMDGPU_INC} -DNAME=kDeviceProfilerAMDGPU
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake
    DEPENDS ${POSEIDON_DEVICE_PROFILER_AMDGPU_BC}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake
    VERBATIM)
  add_custom_target(poseidon_device_profiler_amdgpu
                    DEPENDS ${POSEIDON_DEVICE_PROFILER_AMDGPU_INC})
  add_dependencies(${POSEIDON_PLUGIN_TARGET} poseidon_device_profiler_amdgpu)
  target_include_directories(${POSEIDON_PLUGIN_TARGET}
                             PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  set_property(
    SOURCE DeviceProfile.cpp APPEND
    PROPERTY COMPILE_DEFINITIONS POSEIDON_DEVICE_PROFILER_AMDGPU
             POSEIDON_DEVICE_PROFILER_AMDGPU_ARCH="${POSEIDON_AMDGPU_ARCH}")
  set_property(
    SOURCE DeviceProfile.cpp APPEND
    PROPERTY OBJECT_DEPENDS ${POSEIDON_DEVICE_PROFILER_AMDGPU_INC})
endif()

add_library(poseidon_profile STATIC runtime/fpprofiler/FPProfiler.cpp)
target_include_directories(
  poseidon_profile
  PUBLIC
    $<BUILD_INTERFACE:${ENZYME_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler>
    $<INSTALL_INTERFACE:include>)
target_compile_features(poseidon_profile PRIVATE cxx_std_17)
# Linked into profile-generate builds only: it replaces malloc for the process.
add_library(poseidon_profile_host STATIC runtime/fpprofiler/FPProfilerHost.cpp)
target_compile_features(poseidon_profile_host PRIVATE cxx_std_17)

option(
  POSEIDON_TCEC_USE_CUMPSGEMM
  "Delegate the TCEC runtime to cuMpSGEMM instead of the built-in kernels" OFF)
set(POSEIDON_CUMPSGEMM_ROOT ""
    CACHE
      PATH
      "cuMpSGEMM checkout the TCEC runtime is built against and applications link
     against; required by POSEIDON_TCEC_USE_CUMPSGEMM")

if(CMAKE_CUDA_COMPILER)
  add_library(
    poseidon_rt
    STATIC
      runtime/ozaki/ozaki_rt.cu runtime/tcec/tcec_rt.cu
      runtime/direct/direct_rt.cu)
  # Whole-program per translation unit: every kernel is launched from the host
  # entry point in its own file, so the archive carries complete fatbins and an
  # application built by any compiler can link it.
  set_target_properties(poseidon_rt PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_link_libraries(poseidon_rt PUBLIC CUDA::cublas CUDA::cudart)
  if(POSEIDON_TCEC_USE_CUMPSGEMM)
    if(NOT POSEIDON_CUMPSGEMM_ROOT)
      message(
        FATAL_ERROR "POSEIDON_TCEC_USE_CUMPSGEMM needs POSEIDON_CUMPSGEMM_ROOT")
    endif()
    target_compile_definitions(poseidon_rt PRIVATE POSEIDON_TCEC_USE_CUMPSGEMM)
    target_include_directories(
      poseidon_rt
      PRIVATE ${POSEIDON_CUMPSGEMM_ROOT}/include)
  endif()

  # The host half of the CUDA profiler, linked by every profile-generate build
  # of CUDA code; whole-program for the same reason as poseidon_rt. It takes
  # the program's own CUDA runtime, the one the program's device modules are
  # registered with, so it links none itself.
  add_library(poseidon_profile_cuda STATIC
              runtime/fpprofiler/FPProfilerCUDA.cu)
  set_target_properties(poseidon_profile_cuda
                        PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_include_directories(
    poseidon_profile_cuda
    PRIVATE ${ENZYME_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler)
  target_compile_features(poseidon_profile_cuda PRIVATE cxx_std_17)

  add_executable(poseidon-microbm tools/calibrate/microbm.cu)
  set(POSEIDON_TOOL_TARGETS poseidon-microbm)
  # The dispatch calibrators time the runtime entry points, so they link it.
  foreach(_cal ozaki tcec direct)
    add_executable(poseidon-${_cal}-calibrate
                   tools/calibrate/${_cal}_calibrate.cu)
    target_link_libraries(
      poseidon-${_cal}-calibrate
      PRIVATE poseidon_rt CUDA::cublas)
    if(POSEIDON_TCEC_USE_CUMPSGEMM)
      target_link_directories(
        poseidon-${_cal}-calibrate
        PRIVATE ${POSEIDON_CUMPSGEMM_ROOT}/build)
      target_link_libraries(poseidon-${_cal}-calibrate PRIVATE cumpsgemm)
      set_target_properties(
        poseidon-${_cal}-calibrate
        PROPERTIES BUILD_RPATH ${POSEIDON_CUMPSGEMM_ROOT}/build)
    endif()
    list(APPEND POSEIDON_TOOL_TARGETS poseidon-${_cal}-calibrate)
  endforeach()
endif()

# The same runtimes compiled as HIP for POSEIDON_AMDGPU_ARCH, each source whole
# as the CUDA ones are, by the clang the plugin is built against. TCEC has no
# HIP build.
set(POSEIDON_HIP_RUNTIMES OFF)
if(POSEIDON_ROCM_PATH AND "AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(POSEIDON_HIP_RUNTIMES ON)
  set(POSEIDON_HIP_RUNTIME_HEADERS
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/hip_as_cuda.h
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler/FPProfilerCUDA.h
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler/FPProfileName.h
      ${CMAKE_CURRENT_SOURCE_DIR}/OzakiII.h
      ${ENZYME_SOURCE_DIR}/include/poseidon/poseidon.h)
  function(poseidon_add_hip_library name)
    set(_objs)
    foreach(_src ${ARGN})
      get_filename_component(_stem ${_src} NAME_WE)
      set(_obj ${CMAKE_CURRENT_BINARY_DIR}/hip/${name}/${_stem}.o)
      # HIP marks its error type nodiscard; these sources drop the same results
      # their CUDA builds drop. The HIP headers define CUDA's directed-rounding
      # intrinsics (the Ozaki-II scaling's __dadd_ru, __fsub_rd, ...) only
      # under OCML_BASIC_ROUNDED_OPERATIONS.
      add_custom_command(
        OUTPUT ${_obj}
        COMMAND
          ${LLVM_TOOLS_BINARY_DIR}/clang++ -x hip
          --offload-arch=${POSEIDON_AMDGPU_ARCH}
          --rocm-path=${POSEIDON_ROCM_PATH} -std=c++17 -O3 -fPIC
          -Wno-unused-value -DOCML_BASIC_ROUNDED_OPERATIONS
          -I${ENZYME_SOURCE_DIR}/include
          -I${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler -c
          ${CMAKE_CURRENT_SOURCE_DIR}/${_src} -o ${_obj}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_src}
                ${POSEIDON_HIP_RUNTIME_HEADERS}
        VERBATIM)
      list(APPEND _objs ${_obj})
    endforeach()
    add_library(${name} STATIC ${_objs})
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)
  endfunction()

  # Like poseidon_profile_cuda, it takes the program's own HIP runtime.
  poseidon_add_hip_library(poseidon_profile_hip
                           runtime/fpprofiler/FPProfilerCUDA.cu)
  poseidon_add_hip_library(poseidon_rt_hip runtime/ozaki/ozaki_rt.cu
                           runtime/direct/direct_rt.cu)
  target_link_libraries(
    poseidon_rt_hip INTERFACE ${POSEIDON_ROCM_PATH}/lib/libhipblas.so
                              ${POSEIDON_ROCM_PATH}/lib/libamdhip64.so)

  # The calibrators of an AMD GPU, built the same way. The HIP runtime's own
  # dependencies (the HSA runtime) resolve when a calibrator runs, so a host
  # without the GPU stack links them too.
  function(poseidon_add_hip_tool name src)
    set(_exe ${CMAKE_CURRENT_BINARY_DIR}/bin/${name})
    set(_libs)
    if(ARGN)
      set(_libs -x none ${ARGN})
    endif()
    add_custom_command(
      OUTPUT ${_exe}
      COMMAND
        ${LLVM_TOOLS_BINARY_DIR}/clang++ -x hip
        --offload-arch=${POSEIDON_AMDGPU_ARCH}
        --rocm-path=${POSEIDON_ROCM_PATH} -std=c++17 -O3 -Wno-unused-value
        ${CMAKE_CURRENT_SOURCE_DIR}/${src} ${_libs}
        -L${POSEIDON_ROCM_PATH}/lib -lamdhip64
        -Wl,-rpath,${POSEIDON_ROCM_PATH}/lib -Wl,--allow-shlib-undefined -o
        ${_exe}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src} ${POSEIDON_HIP_RUNTIME_HEADERS}
              ${ARGN}
      VERBATIM)
    add_custom_target(${name} ALL DEPENDS ${_exe})
  endfunction()

  poseidon_add_hip_tool(poseidon-microbm-hip tools/calibrate/microbm_hip.hip)
  set(POSEIDON_HIP_TOOL_TARGETS poseidon-microbm-hip)
  # The dispatch calibrators time hipBLAS; a ROCm installation without it gets
  # the per-op calibrator alone.
  if(EXISTS ${POSEIDON_ROCM_PATH}/lib/libhipblas.so)
    foreach(_cal ozaki direct)
      poseidon_add_hip_tool(
        poseidon-${_cal}-calibrate-hip tools/calibrate/${_cal}_calibrate.cu
        $<TARGET_FILE:poseidon_rt_hip> ${POSEIDON_ROCM_PATH}/lib/libhipblas.so)
      add_dependencies(poseidon-${_cal}-calibrate-hip poseidon_rt_hip)
      list(APPEND POSEIDON_HIP_TOOL_TARGETS poseidon-${_cal}-calibrate-hip)
    endforeach()
  else()
    message(
      STATUS
        "Poseidon: no hipBLAS in ${POSEIDON_ROCM_PATH}; the HIP dispatch calibrators are not built"
    )
  endif()
endif()

add_executable(poseidon-calibrate tools/calibrate/poseidon-calibrate.cpp)
target_compile_definitions(
  poseidon-calibrate
  PRIVATE
    POSEIDON_CALIBRATE_CUMPSGEMM="${POSEIDON_CUMPSGEMM_ROOT}"
    POSEIDON_CALIBRATE_HERBIE="${POSEIDON_HERBIE_BINARY_PATH}"
    POSEIDON_CALIBRATE_CUDA_BIN="${CUDAToolkit_BIN_DIR}"
    POSEIDON_CALIBRATE_PYTHON="${Python3_EXECUTABLE}"
    POSEIDON_CALIBRATE_SRCDIR="${CMAKE_CURRENT_SOURCE_DIR}")
llvm_map_components_to_libnames(POSEIDON_CALIBRATE_LLVM_LIBS support)
target_link_libraries(
  poseidon-calibrate
  PRIVATE ${POSEIDON_CALIBRATE_LLVM_LIBS})
if(CMAKE_CUDA_COMPILER)
  add_dependencies(poseidon-calibrate poseidon-clang ${POSEIDON_TOOL_TARGETS})
endif()
if(POSEIDON_HIP_RUNTIMES)
  add_dependencies(poseidon-calibrate ${POSEIDON_HIP_TOOL_TARGETS})
endif()

# The Enzyme plugin of this build tree differentiates the profiling probes.
set(POSEIDON_ENZYME_PLUGIN "")
if(TARGET ClangEnzyme-${LLVM_VERSION_MAJOR})
  set(POSEIDON_ENZYME_PLUGIN "$<TARGET_FILE:ClangEnzyme-${LLVM_VERSION_MAJOR}>")
endif()

# One binary, two names: the C or the C++ clang, chosen from argv[0].
add_executable(poseidon-clang tools/driver/poseidon-clang.cpp)
target_include_directories(poseidon-clang PRIVATE ${ENZYME_SOURCE_DIR}/include)
target_compile_definitions(
  poseidon-clang
  PRIVATE
    POSEIDON_DRIVER_CLANG="${LLVM_TOOLS_BINARY_DIR}/clang"
    POSEIDON_DRIVER_CLANGXX="${LLVM_TOOLS_BINARY_DIR}/clang++"
    POSEIDON_DRIVER_PLUGIN_NAME="${POSEIDON_PLUGIN_TARGET}${CMAKE_SHARED_MODULE_SUFFIX}"
    POSEIDON_DRIVER_PLUGIN="$<TARGET_FILE:${POSEIDON_PLUGIN_TARGET}>"
    POSEIDON_DRIVER_ENZYME="${POSEIDON_ENZYME_PLUGIN}"
    POSEIDON_DRIVER_INCLUDE="${ENZYME_SOURCE_DIR}/include"
    POSEIDON_DRIVER_LIBDIR="${CMAKE_CURRENT_BINARY_DIR}/lib"
    POSEIDON_DRIVER_PROFILER_CUDA=$<BOOL:${CMAKE_CUDA_COMPILER}>
    POSEIDON_DRIVER_PROFILER_HIP=$<BOOL:${POSEIDON_HIP_RUNTIMES}>
    POSEIDON_DRIVER_CUMPSGEMM="${POSEIDON_CUMPSGEMM_ROOT}"
    POSEIDON_DRIVER_CUDA_ROOT="${CUDAToolkit_TARGET_DIR}"
    POSEIDON_DRIVER_ROCM_ROOT="${POSEIDON_ROCM_PATH}")
llvm_map_components_to_libnames(POSEIDON_DRIVER_LLVM_LIBS support)
target_link_libraries(poseidon-clang PRIVATE ${POSEIDON_DRIVER_LLVM_LIBS})
add_dependencies(poseidon-clang ${POSEIDON_PLUGIN_TARGET})
if(TARGET ClangEnzyme-${LLVM_VERSION_MAJOR})
  add_dependencies(poseidon-clang ClangEnzyme-${LLVM_VERSION_MAJOR})
endif()

# poseidon-clang++ is the same binary; the driver reads its own name.
add_custom_command(
  TARGET poseidon-clang
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink poseidon-clang
    $<TARGET_FILE_DIR:poseidon-clang>/poseidon-clang++)

install(
  TARGETS ${POSEIDON_PLUGIN_TARGET} poseidon_profile poseidon_profile_host
  poseidon-clang poseidon-calibrate
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
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/bin/${_tool}
            DESTINATION ${CMAKE_INSTALL_BINDIR})
  endforeach()
endif()
install(
  DIRECTORY ${ENZYME_SOURCE_DIR}/include/poseidon
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY runtime DESTINATION ${CMAKE_INSTALL_DATADIR}/poseidon)
install(DIRECTORY cost_models DESTINATION ${CMAKE_INSTALL_DATADIR}/poseidon)
install(DIRECTORY tools DESTINATION ${CMAKE_INSTALL_DATADIR}/poseidon)
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
set(POSEIDON_ENZYME_PLUGIN_NAME
    ClangEnzyme-${LLVM_VERSION_MAJOR}${CMAKE_SHARED_MODULE_SUFFIX})
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
set(POSEIDON_CONFIG_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(POSEIDON_CONFIG_ENZYME_PLUGIN
    ${ENZYME_BINARY_DIR}/Enzyme/${POSEIDON_ENZYME_PLUGIN_NAME})
configure_package_config_file(
  cmake/PoseidonConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/PoseidonConfig.cmake
  INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  PATH_VARS POSEIDON_CONFIG_INCLUDE_DIR POSEIDON_CONFIG_DATA_DIR
            POSEIDON_CONFIG_ENZYME_PLUGIN)
# Enzyme installs its plugins to lib/.
set(POSEIDON_CONFIG_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(POSEIDON_CONFIG_DATA_DIR ${CMAKE_INSTALL_DATADIR}/poseidon)
set(POSEIDON_CONFIG_ENZYME_PLUGIN lib/${POSEIDON_ENZYME_PLUGIN_NAME})
configure_package_config_file(
  cmake/PoseidonConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PoseidonConfig.cmake
  INSTALL_DESTINATION ${POSEIDON_INSTALL_CMAKEDIR}
  PATH_VARS POSEIDON_CONFIG_INCLUDE_DIR POSEIDON_CONFIG_DATA_DIR
            POSEIDON_CONFIG_ENZYME_PLUGIN)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/PoseidonConfigVersion.cmake
  VERSION ${POSEIDON_VERSION}
  COMPATIBILITY AnyNewerVersion)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PoseidonConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/PoseidonConfigVersion.cmake
  DESTINATION ${POSEIDON_INSTALL_CMAKEDIR})
