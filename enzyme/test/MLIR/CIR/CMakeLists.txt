# Run regression and unit tests
add_lit_testsuite(
  check-enzymemlir-cir
  "Running MLIR batch mode tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  enzymemlir-opt
  ARGS
  -v)

set_target_properties(check-enzymemlir-cir PROPERTIES FOLDER "Tests")
