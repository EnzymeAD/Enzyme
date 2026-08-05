# These tests are driven by the plugins rather than by enzymemlir-opt, and lit
# gates them on the fir_enzyme_plugin / flang_enzyme_mlir / enzyme_module
# features, so they simply do not run unless the corresponding targets were
# built. Depend on whichever exist in this configuration.
set(ENZYME_FIR_TEST_DEPS)
if (TARGET FIREnzyme-${LLVM_VERSION_MAJOR})
    list(APPEND ENZYME_FIR_TEST_DEPS FIREnzyme-${LLVM_VERSION_MAJOR})
endif()
if (TARGET FlangEnzymeMLIR-${LLVM_VERSION_MAJOR})
    list(APPEND ENZYME_FIR_TEST_DEPS FlangEnzymeMLIR-${LLVM_VERSION_MAJOR})
endif()
# EnzymeFortran (ENZYME_FORTRAN) compiles enzyme.mod, which the tests that
# `use enzyme` include from the module directory.
if (TARGET EnzymeFortran)
    list(APPEND ENZYME_FIR_TEST_DEPS EnzymeFortran)
endif()

# A testsuite of its own so CI can run just the Fortran path without the rest of
# check-enzymemlir.
add_lit_testsuite(check-enzymemlir-fir "Running MLIR FIR/HLFIR tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_FIR_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-enzymemlir-fir PROPERTIES FOLDER "Tests")
