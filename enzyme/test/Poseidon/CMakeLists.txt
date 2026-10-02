set(POSEIDON_TEST_GPU_ARCH "" CACHE STRING
    "CUDA architecture (sm_XY) the integration tests compile for; empty = detect with nvidia-smi")
set(POSEIDON_TEST_COST_MODEL "${HERBIE_PLATFORM_CSV}" CACHE FILEPATH
    "Cost-model CSV measured on the GPU the CUDA tests run on")

set(POSEIDON_TEST_GPU_ARCH_RESOLVED "${POSEIDON_TEST_GPU_ARCH}")
if(NOT POSEIDON_TEST_GPU_ARCH_RESOLVED)
  set(POSEIDON_TEST_GPU_ARCH_RESOLVED "sm_120")
  find_program(POSEIDON_NVIDIA_SMI nvidia-smi)
  if(POSEIDON_NVIDIA_SMI)
    execute_process(COMMAND ${POSEIDON_NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
                    OUTPUT_VARIABLE _cc RESULT_VARIABLE _cc_rc
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_cc_rc EQUAL 0 AND _cc MATCHES "^([0-9]+)\\.([0-9]+)")
      set(POSEIDON_TEST_GPU_ARCH_RESOLVED "sm_${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
    endif()
  endif()
endif()
message(STATUS "Poseidon: integration tests target ${POSEIDON_TEST_GPU_ARCH_RESOLVED}")

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

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(POSEIDON_TEST_DEPS Poseidon-${LLVM_VERSION_MAJOR} poseidon_profile poseidon-clang
    LLVMEnzyme-${LLVM_VERSION_MAJOR})
if(TARGET ClangEnzyme-${LLVM_VERSION_MAJOR})
  list(APPEND POSEIDON_TEST_DEPS ClangEnzyme-${LLVM_VERSION_MAJOR})
endif()
if(TARGET poseidon_rt)
  list(APPEND POSEIDON_TEST_DEPS poseidon_rt)
endif()

# check-poseidon needs no device; the tests under Integration/ compile and
# launch CUDA kernels and have their own target.
add_lit_testsuite(check-poseidon "Running Poseidon regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${POSEIDON_TEST_DEPS}
    ARGS -v --filter-out "Integration/")
set_target_properties(check-poseidon PROPERTIES FOLDER "Tests")

add_subdirectory(Integration)
