# Run regression and unit tests
add_lit_testsuite(check-poseidon-integration "Running Poseidon integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS} ClangEnzyme-${LLVM_VERSION_MAJOR}
    ARGS -v
)

set_target_properties(check-poseidon-integration PROPERTIES FOLDER "Tests")
