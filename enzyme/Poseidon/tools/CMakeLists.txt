# The programs a user runs: the driver poseidon-clang, the cost-model
# calibrator poseidon-calibrate with the GPU programs it runs, the
# condition-number run poseidon-kappa, the explanation reader poseidon-explain,
# and the Herbie the pass runs by default.

include(Herbie)

# --- poseidon-clang

poseidon_gemm_cuda_link_files(_rt_libs)
# Comma-separated: a compile definition is no CMake list.
string(JOIN "," POSEIDON_DRIVER_GEMM_LIBS ${_rt_libs})

# One binary, two names: the C or the C++ clang, chosen from argv[0].
set(LLVM_LINK_COMPONENTS Option Support TargetParser)
add_llvm_executable(poseidon-clang driver/poseidon-clang.cpp)
target_include_directories(
  poseidon-clang
  PRIVATE ${ENZYME_SOURCE_DIR}/include ${POSEIDON_SOURCE_DIR})
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
    POSEIDON_DRIVER_LIBDIR="${POSEIDON_BINARY_DIR}/lib"
    POSEIDON_DRIVER_BINDIR="$<TARGET_FILE_DIR:poseidon-clang>"
    POSEIDON_DRIVER_INSTALL_LIBDIR="${CMAKE_INSTALL_LIBDIR}"
    POSEIDON_DRIVER_INSTALL_INCLUDEDIR="${CMAKE_INSTALL_INCLUDEDIR}"
    POSEIDON_DRIVER_PROFILER_CUDA=$<BOOL:${POSEIDON_HAS_CUDA}>
    POSEIDON_DRIVER_PROFILER_HIP=$<BOOL:${POSEIDON_HAS_HIP}>
    POSEIDON_DRIVER_GEMM_LIBS="${POSEIDON_DRIVER_GEMM_LIBS}"
    POSEIDON_DRIVER_CUDA_ROOT="${POSEIDON_CUDA_ROOT}"
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
  TARGETS poseidon-clang
  EXPORT PoseidonTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(
  CODE
  "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
             poseidon-clang
             \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/poseidon-clang++\")")

# --- poseidon-calibrate

# The GPU programs it runs, one per GPU runtime of the build: the operation
# microbenchmark, and the library calibrator, which times the GEMM runtimes'
# entry points and so links them.
set(_calibrate_gpu_tools)
if(POSEIDON_HAS_CUDA)
  add_executable(poseidon-calibrate-ops-cuda calibrate/ops_gpu.cu)
  add_executable(poseidon-calibrate-library-cuda calibrate/library.cu)
  target_compile_definitions(
    poseidon-calibrate-library-cuda
    PRIVATE POSEIDON_TCEC=$<BOOL:${POSEIDON_HAS_TCEC}>)
  target_include_directories(
    poseidon-calibrate-library-cuda
    PRIVATE ${POSEIDON_SOURCE_DIR})
  target_link_libraries(
    poseidon-calibrate-library-cuda
    PRIVATE poseidon_gemm_cuda CUDA::cublas)
  # Installed, it still finds cuMpSGEMM where it was built, as the programs
  # poseidon-clang links do.
  if(POSEIDON_HAS_TCEC)
    set_target_properties(
      poseidon-calibrate-library-cuda
      PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
  endif()
  install(
    TARGETS poseidon-calibrate-ops-cuda poseidon-calibrate-library-cuda
    EXPORT PoseidonTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  list(APPEND _calibrate_gpu_tools poseidon-clang poseidon-calibrate-ops-cuda
       poseidon-calibrate-library-cuda)
endif()
# poseidon_add_hip_tool(<name> <source> [<library>...])
# The program <name> compiled from <source> as HIP by the plugin's clang and
# linked with the libraries. The HIP runtime's own dependencies (the HSA
# runtime) resolve when a calibrator runs, so a host without the GPU stack
# links them too.
function(poseidon_add_hip_tool name src)
  set(_exe ${POSEIDON_BINARY_DIR}/bin/${name})
  set(_libs)
  if(ARGN)
    set(_libs -x none ${ARGN})
  endif()
  add_custom_command(
    OUTPUT ${_exe}
    COMMAND
      ${POSEIDON_HIP_CXX}
      -I${POSEIDON_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/${src}
      ${_libs}
      -L${POSEIDON_ROCM_PATH}/lib
      -lamdhip64
      -Wl,-rpath,${POSEIDON_ROCM_PATH}/lib
      -Wl,--allow-shlib-undefined
      -o
      ${_exe}
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/${src} ${POSEIDON_RUNTIME_HEADERS} ${ARGN}
    VERBATIM)
  add_custom_target(${name} ALL DEPENDS ${_exe})
  install(PROGRAMS ${_exe} DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()
if(POSEIDON_HAS_HIP)
  poseidon_add_hip_tool(poseidon-calibrate-ops-hip calibrate/ops_gpu.cu)
  poseidon_add_hip_tool(
    poseidon-calibrate-library-hip calibrate/library.cu
    $<TARGET_FILE:poseidon_gemm_hip> ${POSEIDON_ROCM_PATH}/lib/libhipblas.so)
  add_dependencies(poseidon-calibrate-library-hip poseidon_gemm_hip)
  list(APPEND _calibrate_gpu_tools poseidon-calibrate-ops-hip
       poseidon-calibrate-library-hip)
endif()

set(LLVM_LINK_COMPONENTS Support TargetParser)
add_llvm_executable(poseidon-calibrate calibrate/poseidon-calibrate.cpp
                    $<TARGET_OBJECTS:poseidon_costtable>)
target_include_directories(poseidon-calibrate PRIVATE ${POSEIDON_SOURCE_DIR})
target_compile_definitions(
  poseidon-calibrate
  PRIVATE
    POSEIDON_CALIBRATE_TCEC=$<BOOL:${POSEIDON_HAS_TCEC}>
    POSEIDON_CALIBRATE_HERBIE="${POSEIDON_HERBIE_BINARY_PATH}"
    POSEIDON_CALIBRATE_PYTHON="${Python3_EXECUTABLE}"
    POSEIDON_CALIBRATE_SRCDIR="${POSEIDON_SOURCE_DIR}")
if(_calibrate_gpu_tools)
  add_dependencies(poseidon-calibrate ${_calibrate_gpu_tools})
endif()
install(
  TARGETS poseidon-calibrate
  EXPORT PoseidonTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# The sources poseidon-calibrate runs.
install(
  FILES calibrate/inkernel.cu calibrate/ops_cpu.py calibrate/herbie_platform.py
  DESTINATION ${CMAKE_INSTALL_DATADIR}/poseidon/tools/calibrate)

# --- poseidon-kappa and poseidon-explain

add_executable(poseidon-kappa kappa/poseidon-kappa.cpp)
target_include_directories(
  poseidon-kappa
  PRIVATE ${POSEIDON_SOURCE_DIR} ${ENZYME_SOURCE_DIR}/include)
install(
  TARGETS poseidon-kappa
  EXPORT PoseidonTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(PROGRAMS explain/poseidon-explain DESTINATION ${CMAKE_INSTALL_BINDIR})
