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
    Canonicalize.cpp
    CostModel.cpp
    Driver.cpp
    Evaluators.cpp
    Expansion.cpp
    Flags.cpp
    Herbie.cpp
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
    StageParam.cpp
    Staging.cpp
    Types.cpp
    Utils.cpp
    WmmaUtils.cpp
    matmul/MatmulAccuracy.cpp
    matmul/MatmulApply.cpp
    matmul/MatmulCandidates.cpp
    matmul/MatmulPricing.cpp
    matmul/MatmulProfile.cpp
    matmul/RecognizeHostGemm.cpp
)

set(POSEIDON_PLUGIN_TARGET Poseidon-${LLVM_VERSION_MAJOR})
if(Clang_FOUND)
  add_llvm_library(${POSEIDON_PLUGIN_TARGET} ${POSEIDON_SRC}
                   MODULE PARTIAL_SOURCES_INTENDED PLUGIN_TOOL clang)
else()
  add_llvm_library(${POSEIDON_PLUGIN_TARGET} ${POSEIDON_SRC}
                   MODULE PARTIAL_SOURCES_INTENDED PLUGIN_TOOL opt)
endif()
target_include_directories(${POSEIDON_PLUGIN_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENZYME_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler)
target_link_libraries(${POSEIDON_PLUGIN_TARGET} PRIVATE MPFR::MPFR)
set_source_files_properties(Herbie.cpp PROPERTIES
    COMPILE_DEFINITIONS HERBIE_BINARY="${POSEIDON_HERBIE_BINARY_PATH}")
include(GNUInstallDirs)
# Where a compile looks for the cost model of its target device when neither
# the flag nor the environment names one.
set_source_files_properties(CostModel.cpp PROPERTIES
    COMPILE_DEFINITIONS
    POSEIDON_INSTALL_COST_MODELS="${CMAKE_INSTALL_FULL_DATADIR}/poseidon/cost_models")

add_library(poseidon_profile STATIC runtime/fpprofiler/FPProfiler.cpp)
target_include_directories(poseidon_profile PUBLIC
    $<BUILD_INTERFACE:${ENZYME_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler>
    $<INSTALL_INTERFACE:include>)
target_compile_features(poseidon_profile PRIVATE cxx_std_17)

option(POSEIDON_TCEC_USE_CUMPSGEMM
       "Delegate the TCEC runtime to cuMpSGEMM instead of the built-in kernels" OFF)
set(POSEIDON_CUMPSGEMM_ROOT "" CACHE PATH
    "cuMpSGEMM checkout the TCEC runtime is built against and applications link
     against; required by POSEIDON_TCEC_USE_CUMPSGEMM")

if(CMAKE_CUDA_COMPILER)
  add_library(poseidon_rt STATIC
      runtime/ozaki/ozaki_rt.cu
      runtime/tcec/tcec_rt.cu
      runtime/direct/direct_rt.cu
      runtime/stage/stage_rt.cu)
  # Whole-program per translation unit: every kernel is launched from the host
  # entry point in its own file, so the archive carries complete fatbins and an
  # application built by any compiler can link it.
  set_target_properties(poseidon_rt PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_link_libraries(poseidon_rt PUBLIC CUDA::cublas CUDA::cudart)
  if(POSEIDON_TCEC_USE_CUMPSGEMM)
    if(NOT POSEIDON_CUMPSGEMM_ROOT)
      message(FATAL_ERROR "POSEIDON_TCEC_USE_CUMPSGEMM needs POSEIDON_CUMPSGEMM_ROOT")
    endif()
    target_compile_definitions(poseidon_rt PRIVATE POSEIDON_TCEC_USE_CUMPSGEMM)
    target_include_directories(poseidon_rt PRIVATE ${POSEIDON_CUMPSGEMM_ROOT}/include)
  endif()

  add_executable(poseidon-microbm tools/calibrate/microbm.cu)
  set(POSEIDON_TOOL_TARGETS poseidon-microbm)
  # The dispatch calibrators time the runtime entry points, so they link it.
  foreach(_cal ozaki tcec direct)
    add_executable(poseidon-${_cal}-calibrate tools/calibrate/${_cal}_calibrate.cu)
    target_link_libraries(poseidon-${_cal}-calibrate PRIVATE poseidon_rt CUDA::cublas)
    if(POSEIDON_TCEC_USE_CUMPSGEMM)
      target_link_directories(poseidon-${_cal}-calibrate PRIVATE
                              ${POSEIDON_CUMPSGEMM_ROOT}/build)
      target_link_libraries(poseidon-${_cal}-calibrate PRIVATE cumpsgemm)
      set_target_properties(poseidon-${_cal}-calibrate PROPERTIES
          BUILD_RPATH ${POSEIDON_CUMPSGEMM_ROOT}/build)
    endif()
    list(APPEND POSEIDON_TOOL_TARGETS poseidon-${_cal}-calibrate)
  endforeach()
endif()

add_executable(poseidon-calibrate tools/calibrate/poseidon-calibrate.cpp)
target_compile_definitions(poseidon-calibrate PRIVATE
    POSEIDON_CALIBRATE_CUMPSGEMM="${POSEIDON_CUMPSGEMM_ROOT}"
    POSEIDON_CALIBRATE_HERBIE="${POSEIDON_HERBIE_BINARY_PATH}"
    POSEIDON_CALIBRATE_CUDA_BIN="${CUDAToolkit_BIN_DIR}"
    POSEIDON_CALIBRATE_PYTHON="${Python3_EXECUTABLE}"
    POSEIDON_CALIBRATE_SRCDIR="${CMAKE_CURRENT_SOURCE_DIR}"
    POSEIDON_CALIBRATE_INKERNEL_PROFILE="${CMAKE_CURRENT_SOURCE_DIR}/artifacts/cgo2027/benchmarks/ozp/fpprofile")
llvm_map_components_to_libnames(POSEIDON_CALIBRATE_LLVM_LIBS support)
target_link_libraries(poseidon-calibrate PRIVATE ${POSEIDON_CALIBRATE_LLVM_LIBS})
if(CMAKE_CUDA_COMPILER)
  add_dependencies(poseidon-calibrate poseidon-clang ${POSEIDON_TOOL_TARGETS})
endif()

# The Enzyme plugin of this build tree differentiates the profiling probes.
set(POSEIDON_ENZYME_PLUGIN "")
if(TARGET ClangEnzyme-${LLVM_VERSION_MAJOR})
  set(POSEIDON_ENZYME_PLUGIN "$<TARGET_FILE:ClangEnzyme-${LLVM_VERSION_MAJOR}>")
endif()

# One binary, two names: the C or the C++ clang, chosen from argv[0].
add_executable(poseidon-clang tools/driver/poseidon-clang.cpp)
target_include_directories(poseidon-clang PRIVATE ${ENZYME_SOURCE_DIR}/include)
target_compile_definitions(poseidon-clang PRIVATE
    POSEIDON_DRIVER_CLANG="${LLVM_TOOLS_BINARY_DIR}/clang"
    POSEIDON_DRIVER_CLANGXX="${LLVM_TOOLS_BINARY_DIR}/clang++"
    POSEIDON_DRIVER_PLUGIN_NAME="${POSEIDON_PLUGIN_TARGET}${CMAKE_SHARED_MODULE_SUFFIX}"
    POSEIDON_DRIVER_PLUGIN="$<TARGET_FILE:${POSEIDON_PLUGIN_TARGET}>"
    POSEIDON_DRIVER_ENZYME="${POSEIDON_ENZYME_PLUGIN}"
    POSEIDON_DRIVER_INCLUDE="${ENZYME_SOURCE_DIR}/include"
    POSEIDON_DRIVER_LIBDIR="${CMAKE_CURRENT_BINARY_DIR}/lib"
    POSEIDON_DRIVER_PROFILER_CU="${CMAKE_CURRENT_SOURCE_DIR}/runtime/fpprofiler/FPProfilerCUDA.cu"
    POSEIDON_DRIVER_CUMPSGEMM="${POSEIDON_CUMPSGEMM_ROOT}"
    POSEIDON_DRIVER_CUDA_ROOT="${CUDAToolkit_TARGET_DIR}")
llvm_map_components_to_libnames(POSEIDON_DRIVER_LLVM_LIBS support)
target_link_libraries(poseidon-clang PRIVATE ${POSEIDON_DRIVER_LLVM_LIBS})
add_dependencies(poseidon-clang ${POSEIDON_PLUGIN_TARGET})
if(TARGET ClangEnzyme-${LLVM_VERSION_MAJOR})
  add_dependencies(poseidon-clang ClangEnzyme-${LLVM_VERSION_MAJOR})
endif()

# poseidon-clang++ is the same binary; the driver reads its own name.
add_custom_command(TARGET poseidon-clang POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            poseidon-clang $<TARGET_FILE_DIR:poseidon-clang>/poseidon-clang++)

install(TARGETS ${POSEIDON_PLUGIN_TARGET} poseidon_profile poseidon-clang
                poseidon-calibrate
        EXPORT PoseidonTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
             poseidon-clang
             \"\$ENV{DESTDIR}${CMAKE_INSTALL_FULL_BINDIR}/poseidon-clang++\")")
if(CMAKE_CUDA_COMPILER)
  install(TARGETS poseidon_rt ${POSEIDON_TOOL_TARGETS}
          EXPORT PoseidonTargets
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY ${ENZYME_SOURCE_DIR}/include/poseidon DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY runtime DESTINATION ${CMAKE_INSTALL_DATADIR}/poseidon)
install(DIRECTORY cost_models DESTINATION ${CMAKE_INSTALL_DATADIR}/poseidon)
install(DIRECTORY tools DESTINATION ${CMAKE_INSTALL_DATADIR}/poseidon)
install(EXPORT PoseidonTargets NAMESPACE Poseidon::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Poseidon)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/PoseidonConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/PoseidonConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Poseidon)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/PoseidonConfig.cmake
              ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindMPFR.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Poseidon)
