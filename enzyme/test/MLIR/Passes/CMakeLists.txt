# Run regression and unit tests
add_lit_testsuite(check-enzymemlir-pass "Running MLIR pass tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS enzymemlir-opt
    ARGS -v
)

set_target_properties(check-enzymemlir-pass PROPERTIES FOLDER "Tests")
