# Fortran integration tests: complete numerical methods that use Enzyme for
# one piece of derivative information (a Jacobian-vector product, a Jacobian
# column), as opposed to the single-routine tests in enzyme/test/Fortran.
add_lit_testsuite(
  check-enzyme-integration-fortran
  "Running enzyme fortran integration tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  ${ENZYME_TEST_DEPS}
  LLVMEnzyme-${LLVM_VERSION_MAJOR}
  ARGS
  -v)

set_target_properties(check-enzyme-integration-fortran PROPERTIES FOLDER "Tests")
