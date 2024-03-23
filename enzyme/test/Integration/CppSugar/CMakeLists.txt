# Run regression and unit tests
add_lit_testsuite(check-enzyme-integration-sugar "Running enzyme c++ sugar integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS} ClangEnzyme-${LLVM_VERSION_MAJOR}
    ARGS -v
)

set_target_properties(check-enzyme-integration-sugar PROPERTIES FOLDER "Tests")

# add_lit_testsuites(ENZYME ${CMAKE_CURRENT_SOURCE_DIR}
# DEPENDS ${ENZYME_TEST_DEPS}
# )
