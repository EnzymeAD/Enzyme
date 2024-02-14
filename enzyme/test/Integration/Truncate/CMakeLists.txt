add_lit_testsuite(check-enzyme-integration-truncate "Running enzyme fp truncation integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-enzyme-integration-truncate PROPERTIES FOLDER "Tests")

