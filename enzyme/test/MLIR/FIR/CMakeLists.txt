# These tests are driven by the flang plugins, and lit gates them on the
# flang_enzyme_mlir feature, so they do not run unless the plugins were built.
set(ENZYME_FIR_TEST_DEPS)
if(TARGET FlangEnzymeMLIR-${LLVM_VERSION_MAJOR})
  list(APPEND ENZYME_FIR_TEST_DEPS FlangEnzymeMLIR-${LLVM_VERSION_MAJOR})
endif()
if(TARGET FlangEnzyme-${LLVM_VERSION_MAJOR})
  list(APPEND ENZYME_FIR_TEST_DEPS FlangEnzyme-${LLVM_VERSION_MAJOR})
endif()

# A testsuite of its own so CI can run just the Fortran path without the rest of
# check-enzymemlir.
add_lit_testsuite(
  check-enzymemlir-fir
  "Running MLIR FIR tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  ${ENZYME_FIR_TEST_DEPS}
  ARGS
  -v)

set_target_properties(check-enzymemlir-fir PROPERTIES FOLDER "Tests")
