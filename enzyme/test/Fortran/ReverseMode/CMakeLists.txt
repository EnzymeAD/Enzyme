# Run regression and unit tests
add_lit_testsuite(check-enzyme-fortran-reverse "Running enzyme reverse mode fortran integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS} LLVMEnzyme-${LLVM_VERSION_MAJOR}
    ARGS -v
)

set_target_properties(check-enzyme-fortran-reverse PROPERTIES FOLDER "Tests")