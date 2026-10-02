add_lit_testsuite(check-poseidon-integration "Running Poseidon integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${POSEIDON_TEST_DEPS}
    ARGS -v)
set_target_properties(check-poseidon-integration PROPERTIES FOLDER "Tests")
