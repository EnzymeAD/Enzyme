# A test program that builds type trees with MTypeTree and prints a module of
# their #enzyme.type_tree attributes; type_tree_attr.cpp holds its RUN lines.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(enzymemlir-type-tree-test type_tree_attr.cpp)
llvm_update_compile_flags(enzymemlir-type-tree-test)
target_include_directories(
  enzymemlir-type-tree-test
  PRIVATE ${PROJECT_SOURCE_DIR}/Enzyme/MLIR ${PROJECT_BINARY_DIR}/Enzyme/MLIR)
target_link_libraries(
  enzymemlir-type-tree-test
  PRIVATE ${dialect_libs} MLIREnzymeAutoDiffInterface MLIRIR)
