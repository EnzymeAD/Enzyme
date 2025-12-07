# Run regression and unit tests
add_lit_testsuite(check-enzyme-simplegvn "Running enzyme simple-gvn tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-enzyme-simplegvn PROPERTIES FOLDER "Tests")
