# Run regression and unit tests
add_lit_testsuite(check-enzymemlir-optimizead "Running MLIR OptimizeAD tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS enzymemlir-opt
    ARGS -v
)

set_target_properties(check-enzymemlir-optimizead PROPERTIES FOLDER "Tests")
