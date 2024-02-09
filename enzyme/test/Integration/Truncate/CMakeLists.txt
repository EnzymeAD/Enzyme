# Run regression and unit tests
add_lit_testsuite(check-enzyme-integration-truncate "Running enzyme batch mode integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-enzyme-integration-truncate PROPERTIES FOLDER "Tests")

