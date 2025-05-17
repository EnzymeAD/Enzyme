# Run regression and unit tests
add_lit_testsuite(check-enzymemlir-alias "Running MLIR alias mode tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS enzymemlir-opt
    ARGS -v
)

set_target_properties(check-enzymemlir-alias PROPERTIES FOLDER "Tests")
