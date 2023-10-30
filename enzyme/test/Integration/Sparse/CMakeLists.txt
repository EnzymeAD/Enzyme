# Run regression and unit tests
add_lit_testsuite(check-enzyme-integration-sparse "Running enzyme spare integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ClangEnzyme-${LLVM_VERSION_MAJOR}
    ARGS -v
)

set_target_properties(check-enzyme-integration-sparse PROPERTIES FOLDER "Tests")

#add_lit_testsuites(ENZYME ${CMAKE_CURRENT_SOURCE_DIR}
    #    DEPENDS ${ENZYME_TEST_DEPS}
#)
