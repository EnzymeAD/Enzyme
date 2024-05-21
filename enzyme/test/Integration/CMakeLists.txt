add_subdirectory(CppSugar)
add_subdirectory(ForwardMode)
add_subdirectory(ForwardError)
add_subdirectory(ForwardModeVector)
add_subdirectory(ReverseMode)
add_subdirectory(BatchMode)
add_subdirectory(Sparse)
add_subdirectory(Truncate)

# Run regression and unit tests
add_lit_testsuite(check-enzyme-integration "Running enzyme integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS} ClangEnzyme-${LLVM_VERSION_MAJOR}
    ARGS -v
)

set_target_properties(check-enzyme-integration PROPERTIES FOLDER "Tests")
