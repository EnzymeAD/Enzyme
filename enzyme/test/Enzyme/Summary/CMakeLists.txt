# Run regression and unit tests
add_lit_testsuite(
  check-enzyme-summary
  "Running enzyme enzyme-summary tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  ${ENZYME_TEST_DEPS}
  ARGS
  -v)

set_target_properties(check-enzyme-summary PROPERTIES FOLDER "Tests")
