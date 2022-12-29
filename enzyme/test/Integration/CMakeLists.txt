add_subdirectory(ForwardMode)
add_subdirectory(ForwardModeVector)
add_subdirectory(ReverseMode)
add_subdirectory(BatchMode)

# Run regression and unit tests
add_lit_testsuite(check-enzyme-integration "Running enzyme integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS} ClangEnzyme-${LLVM_VERSION_MAJOR}
    ARGS -v
)

add_lit_testsuite(check-enzyme-integration-bench "Benchmarking enzyme integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v --time-tests -o results.json
)

set_target_properties(check-enzyme-integration PROPERTIES FOLDER "Tests")
set_target_properties(check-enzyme-integration-bench PROPERTIES FOLDER "Tests")
