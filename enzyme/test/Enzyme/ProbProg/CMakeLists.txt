# Run regression and unit tests
add_lit_testsuite(check-enzyme-probprog "Running enzyme probabilistic programming mode regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-enzyme-probprog PROPERTIES FOLDER "Tests")

# add_lit_testsuites(ENZYME ${CMAKE_CURRENT_SOURCE_DIR}
# DEPENDS ${ENZYME_TEST_DEPS}
# )
