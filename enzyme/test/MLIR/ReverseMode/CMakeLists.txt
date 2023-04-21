# Run regression and unit tests
add_lit_testsuite(check-enzymemlir-reverse "Running MLIR reverse mode tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS enzymemlir-opt
    ARGS -v
)

set_target_properties(check-enzymemlir-reverse PROPERTIES FOLDER "Tests")
