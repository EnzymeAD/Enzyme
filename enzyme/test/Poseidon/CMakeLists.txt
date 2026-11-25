set(POSEIDON_TEST_GPU_ARCH ""
    CACHE
      STRING
      "CUDA architecture (sm_XY) the integration tests compile for; empty = the CUDA runtimes' first, else sm_120")
set(POSEIDON_TEST_GPU_ARCH_RESOLVED "${POSEIDON_TEST_GPU_ARCH}")
if(NOT POSEIDON_TEST_GPU_ARCH_RESOLVED)
  set(POSEIDON_TEST_GPU_ARCH_RESOLVED "sm_120")
  if(CMAKE_CUDA_COMPILER AND POSEIDON_CUDA_ARCH)
    set(POSEIDON_TEST_GPU_ARCH_RESOLVED "${POSEIDON_CUDA_ARCH}")
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
set(_platform ${POSEIDON_SOURCE_DIR}/tools/calibrate/herbie_platform.py)
set(_outputs "")
foreach(
  _model
  cpu_native
  cpu_foreign
  cpu_noarch
  cpu_missing
  gfx942_synthetic
  sm_120_synthetic_dispatch
  sm_120_synthetic_tcec
  sm_120_synthetic_half)
  list(APPEND _outputs ${POSEIDON_TEST_COST_MODELS}/cm_${_model}.csv)
endforeach()
set(_platforms "")
foreach(_model sm_120_synthetic sm_90_synthetic gpu_synthetic)
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

# Programs tests run: drivers of pass sources, and opt plugins that call the
# Poseidon plugin's functions, which they find in the opt process that loaded
# both. What they print is checked against what clang computes from them by
# default, so assertions stay on in every build type and floating-point
# contraction is clang's default with any compiler. They are built for the
# tests only (poseidon-test-programs), not with the compiler.
set(POSEIDON_TEST_PROGRAM_OPTIONS -UNDEBUG)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  list(APPEND POSEIDON_TEST_PROGRAM_OPTIONS -ffp-contract=on)
endif()
set(POSEIDON_TEST_PROGRAM_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
llvm_map_components_to_libnames(POSEIDON_TEST_PROGRAM_LLVM_LIBS support)

function(poseidon_configure_test_program name)
  target_include_directories(
    ${name}
    PRIVATE ${POSEIDON_SOURCE_DIR} ${ENZYME_SOURCE_DIR}/include)
  target_compile_options(${name} PRIVATE ${POSEIDON_TEST_PROGRAM_OPTIONS})
  set_target_properties(
    ${name}
    PROPERTIES
      EXCLUDE_FROM_ALL ON RUNTIME_OUTPUT_DIRECTORY ${POSEIDON_TEST_PROGRAM_DIR}
      LIBRARY_OUTPUT_DIRECTORY ${POSEIDON_TEST_PROGRAM_DIR})
endfunction()

function(poseidon_add_test_program name)
  add_executable(${name} ${ARGN})
  poseidon_configure_test_program(${name})
  target_link_libraries(${name} PRIVATE ${POSEIDON_TEST_PROGRAM_LLVM_LIBS})
endfunction()

function(poseidon_add_test_plugin name source)
  add_llvm_library(${name} MODULE BUILDTREE_ONLY ${source} PLUGIN_TOOL opt)
  poseidon_configure_test_program(${name})
endfunction()

poseidon_add_test_program(
  poseidon-test-round-to-prec Core/round_to_prec_apfloat.cpp
  ${POSEIDON_SOURCE_DIR}/Core/FPKind.cpp)
poseidon_add_test_program(poseidon-test-matmul-scheme Matmul/matmul_scheme.cpp
                          ${POSEIDON_SOURCE_DIR}/Core/FPKind.cpp)
poseidon_add_test_plugin(poseidon-test-expansion-match
                         Inputs/expansion_match/emit.cpp)
poseidon_add_test_plugin(poseidon-test-model Inputs/model/plugin.cpp)
set(POSEIDON_TEST_PROGRAMS
    poseidon-test-round-to-prec poseidon-test-matmul-scheme
    poseidon-test-expansion-match poseidon-test-model)
add_custom_target(poseidon-test-programs DEPENDS ${POSEIDON_TEST_PROGRAMS})

# The Enzyme plugin comes in two builds: ClangEnzyme for -fpass-plugin and
# LLVMEnzyme for opt's -load-pass-plugin.
set(POSEIDON_ENZYME_OPT_PLUGIN
    "${ENZYME_BINARY_DIR}/Enzyme/LLVMEnzyme-${LLVM_VERSION_MAJOR}${LLVM_SHLIBEXT}")

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
# A test that links poseidon_rt links what the runtime links, in its order.
set(POSEIDON_TEST_RT_LINK "")
poseidon_rt_link_files(_rt_libs TOOLKIT)
foreach(_lib ${_rt_libs})
  get_filename_component(_dir ${_lib} DIRECTORY)
  string(APPEND POSEIDON_TEST_RT_LINK " ${_lib} -Wl,-rpath,${_dir}")
endforeach()
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
    poseidon-test-programs
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
if(TARGET herbie)
  list(APPEND POSEIDON_TEST_DEPS herbie)
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
