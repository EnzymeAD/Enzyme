# Run regression and unit tests
add_lit_testsuite(check-enzyme-fpopt "Running enzyme floating-point optimization regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v -j1
)

set_target_properties(check-enzyme-fpopt PROPERTIES FOLDER "Tests")

#add_lit_testsuites(ENZYME ${CMAKE_CURRENT_SOURCE_DIR}
    #    DEPENDS ${ENZYME_TEST_DEPS}
#)
