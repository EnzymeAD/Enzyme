# Run regression and unit tests
add_lit_testsuite(check-enzymemlir-activity "Running MLIR activity tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS enzymemlir-opt
    ARGS -v
)

set_target_properties(check-enzymemlir-activity PROPERTIES FOLDER "Tests")
