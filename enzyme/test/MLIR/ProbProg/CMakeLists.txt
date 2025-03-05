# Run regression and unit tests
add_lit_testsuite(check-enzymemlir-propprog "Running MLIR probprog tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS enzymemlir-opt
    ARGS -v
)

set_target_properties(check-enzymemlir-propprog PROPERTIES FOLDER "Tests")
