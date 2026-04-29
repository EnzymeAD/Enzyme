# Run regression and unit tests
add_lit_testsuite(check-enzymemlir-impulse "Running MLIR Impulse dialect tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS enzymemlir-opt
    ARGS -v
)

set_target_properties(check-enzymemlir-impulse PROPERTIES FOLDER "Tests")
