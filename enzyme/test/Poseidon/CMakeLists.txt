set(POSEIDON_TEST_GPU_ARCH ""
    CACHE
      STRING
      "CUDA architecture (sm_XY) the integration tests compile for; empty = detect with nvidia-smi")
set(POSEIDON_TEST_GPU_ARCH_RESOLVED "${POSEIDON_TEST_GPU_ARCH}")
if(NOT POSEIDON_TEST_GPU_ARCH_RESOLVED)
  set(POSEIDON_TEST_GPU_ARCH_RESOLVED "sm_120")
  find_program(POSEIDON_NVIDIA_SMI nvidia-smi)
  if(POSEIDON_NVIDIA_SMI)
    execute_process(
      COMMAND
        ${POSEIDON_NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _cc
      RESULT_VARIABLE _cc_rc OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_cc_rc EQUAL 0 AND _cc MATCHES "^([0-9]+)\\.([0-9]+)")
      set(POSEIDON_TEST_GPU_ARCH_RESOLVED "sm_${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
    endif()
  endif()
endif()
message(
  STATUS "Poseidon: integration tests target ${POSEIDON_TEST_GPU_ARCH_RESOLVED}")

# The tests price from made-up cost models written into the build tree. A GPU
# model a test runs Herbie under carries the Herbie platform generated from it,
# as a calibrated model does. No test prices from a measurement.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(POSEIDON_TEST_COST_MODELS ${CMAKE_CURRENT_BINARY_DIR}/cost_models)
set(_generator ${CMAKE_CURRENT_SOURCE_DIR}/Inputs/make_cost_models.py)
set(_platform ${POSEIDON_SOURCE_DIR}/tools/herbie/csv_to_herbie_platform.py)
set(_outputs "")
foreach(
  _model
  cpu_native
  cpu_foreign
  cpu_noarch
  cpu_missing
  gfx942_synthetic
  sm_120_synthetic_dispatch
  sm_120_synthetic_half)
  list(APPEND _outputs ${POSEIDON_TEST_COST_MODELS}/cm_${_model}.csv)
endforeach()
set(_platforms "")
foreach(_arch_model sm_120:sm_120_synthetic sm_90:sm_90_synthetic
        ${POSEIDON_TEST_GPU_ARCH_RESOLVED}:gpu_synthetic)
  string(REPLACE ":" ";" _arch_model ${_arch_model})
  list(GET _arch_model 0 _arch)
  list(GET _arch_model 1 _model)
  set(_csv ${POSEIDON_TEST_COST_MODELS}/cm_${_model}.csv)
  set(_rkt ${POSEIDON_TEST_COST_MODELS}/cm_${_model}.herbie.rkt)
  list(APPEND _outputs ${_csv} ${_rkt})
  list(
    APPEND
    _platforms
    COMMAND
    ${Python3_EXECUTABLE}
    ${_platform}
    --csv
    ${_csv}
    --arch
    ${_arch}
    --output
    ${_rkt})
endforeach()
add_custom_command(
  OUTPUT ${_outputs}
  COMMAND
    ${Python3_EXECUTABLE} ${_generator} ${POSEIDON_TEST_COST_MODELS}
    ${POSEIDON_TEST_GPU_ARCH_RESOLVED} ${_platforms}
  DEPENDS ${_generator} ${_platform}
  COMMENT "Writing the cost models of the Poseidon tests")
add_custom_target(poseidon-test-cost-models DEPENDS ${_outputs})

# The Enzyme plugin comes in two builds: ClangEnzyme for -fpass-plugin and
# LLVMEnzyme for opt's -load-pass-plugin.
set(POSEIDON_ENZYME_CLANG_PLUGIN "")
set(POSEIDON_ENZYME_OPT_PLUGIN
    "${ENZYME_BINARY_DIR}/Enzyme/LLVMEnzyme-${LLVM_VERSION_MAJOR}${LLVM_SHLIBEXT}")
if(TARGET ClangEnzyme-${LLVM_VERSION_MAJOR})
  set(POSEIDON_ENZYME_CLANG_PLUGIN
      "${ENZYME_BINARY_DIR}/Enzyme/ClangEnzyme-${LLVM_VERSION_MAJOR}${LLVM_SHLIBEXT}")
endif()

if(CMAKE_CUDA_COMPILER)
  set(POSEIDON_TEST_HAS_CUDA True)
else()
  set(POSEIDON_TEST_HAS_CUDA False)
endif()
# The CUDA libraries the integration tests link, from the toolkit the build
# found.
find_package(CUDAToolkit QUIET)
if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(POSEIDON_TEST_HAS_AMDGPU True)
else()
  set(POSEIDON_TEST_HAS_AMDGPU False)
endif()
if(TARGET poseidon_rt_hip)
  set(POSEIDON_TEST_HAS_HIP True)
else()
  set(POSEIDON_TEST_HAS_HIP False)
endif()
# A test that links poseidon_rt links what the runtime links, in the
# runtime's order: cuMpSGEMM defines cuBLAS's GEMM entry points, and comes
# after cuBLAS so that a call of cuBLAS reaches cuBLAS.
set(POSEIDON_TEST_RT_LINK "")
if(TARGET poseidon_rt)
  get_target_property(_rt_libs poseidon_rt INTERFACE_LINK_LIBRARIES)
  foreach(_lib ${_rt_libs})
    if(TARGET ${_lib})
      get_target_property(_lib ${_lib} LOCATION)
    endif()
    get_filename_component(_dir ${_lib} DIRECTORY)
    string(APPEND POSEIDON_TEST_RT_LINK " ${_lib} -Wl,-rpath,${_dir}")
  endforeach()
endif()
if(POSEIDON_TCEC_USE_CUMPSGEMM)
  set(POSEIDON_TEST_TCEC_CUMPSGEMM True)
else()
  set(POSEIDON_TEST_TCEC_CUMPSGEMM False)
endif()
# The device profilers the plugin carries.
if(TARGET poseidon_device_profiler)
  set(POSEIDON_TEST_NVPTX_PROFILER True)
else()
  set(POSEIDON_TEST_NVPTX_PROFILER False)
endif()
if(TARGET poseidon_device_profiler_amdgpu)
  set(POSEIDON_TEST_AMDGPU_PROFILER True)
else()
  set(POSEIDON_TEST_AMDGPU_PROFILER False)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(POSEIDON_TEST_DEPS
    Poseidon-${LLVM_VERSION_MAJOR}
    poseidon_profile
    poseidon_profile_host
    poseidon-clang
    poseidon-calibrate
    poseidon-probe
    poseidon-test-cost-models
    LLVMEnzyme-${LLVM_VERSION_MAJOR})
if(TARGET ClangEnzyme-${LLVM_VERSION_MAJOR})
  list(APPEND POSEIDON_TEST_DEPS ClangEnzyme-${LLVM_VERSION_MAJOR})
endif()
if(TARGET poseidon_rt)
  list(APPEND POSEIDON_TEST_DEPS poseidon_rt poseidon_profile_cuda)
endif()
if(TARGET poseidon_rt_hip)
  list(APPEND POSEIDON_TEST_DEPS poseidon_rt_hip poseidon_profile_hip)
endif()

# check-poseidon needs no device; the tests under Integration/ compile and
# launch CUDA kernels and have their own target.
add_lit_testsuite(
  check-poseidon
  "Running Poseidon regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  ${POSEIDON_TEST_DEPS}
  ARGS
  -v
  --filter-out
  "Integration/")
set_target_properties(check-poseidon PROPERTIES FOLDER "Tests")

add_subdirectory(Integration)
