# Run regression and unit tests
add_lit_testsuite(check-enzyme-blas "Running enzyme blas regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-enzyme-blas PROPERTIES FOLDER "Tests")
