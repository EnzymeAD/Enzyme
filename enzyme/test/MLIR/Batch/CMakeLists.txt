# Run regression and unit tests
add_lit_testsuite(check-enzymemlir-batch "Running MLIR batch mode tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS enzymemlir-opt
    ARGS -v
)

set_target_properties(check-enzymemlir-batch PROPERTIES FOLDER "Tests")
