# Run regression and unit tests
add_lit_testsuite(
  check-enzyme-typehints
  "Running enzyme type hint tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  ${ENZYME_TEST_DEPS}
  ARGS
  -v)

set_target_properties(check-enzyme-typehints PROPERTIES FOLDER "Tests")
