# Run regression and unit tests
add_lit_testsuite(
  check-enzyme-fortran-forward-intrinsics
  "Running enzyme forward mode fortran integration tests for intrinsics"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  ${ENZYME_TEST_DEPS}
  LLVMEnzyme-${LLVM_VERSION_MAJOR}
  ARGS
  -v)

set_target_properties(check-enzyme-fortran-forward-intrinsics PROPERTIES FOLDER "Tests")
