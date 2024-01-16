# Run regression and unit tests
add_lit_testsuite(check-enzyme-trunc "Running enzyme truncation tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-enzyme-trunc PROPERTIES FOLDER "Tests")

# add_lit_testsuites(ENZYME ${CMAKE_CURRENT_SOURCE_DIR}
# DEPENDS ${ENZYME_TEST_DEPS}
# )
