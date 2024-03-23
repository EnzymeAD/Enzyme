# Run regression and unit tests
add_lit_testsuite(check-enzyme-integration-forwarderror "Running enzyme forward error propagation integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS} ClangEnzyme-${LLVM_VERSION_MAJOR}
    ARGS -v
)

set_target_properties(check-enzyme-integration-forwarderror PROPERTIES FOLDER "Tests")

#add_lit_testsuites(ENZYME ${CMAKE_CURRENT_SOURCE_DIR}
    #    DEPENDS ${ENZYME_TEST_DEPS}
#)
