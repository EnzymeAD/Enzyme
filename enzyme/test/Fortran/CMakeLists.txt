message("Building Fortran tests")

add_subdirectory(ForwardMode)
add_subdirectory(ReverseMode)

# The FlangEnzyme frontend plugin is exercised by flang_plugin_sketch.f90 (gated
# on the `flang` lit feature); make sure it is built before running the suite.
set(ENZYME_FORTRAN_TEST_DEPS ${ENZYME_TEST_DEPS} LLVMEnzyme-${LLVM_VERSION_MAJOR})
if (ENZYME_FLANG AND Flang_FOUND AND LLVM_VERSION_MAJOR GREATER_EQUAL 20)
    list(APPEND ENZYME_FORTRAN_TEST_DEPS FlangEnzyme-${LLVM_VERSION_MAJOR})
endif()

# Run regression and unit tests
add_lit_testsuite(check-enzyme-fortran "Running enzyme fortran integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_FORTRAN_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-enzyme-fortran PROPERTIES FOLDER "Tests")
