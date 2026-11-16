# FlangEnzymeMLIR: a plugin for `flang -fc1 -load` that adds Enzyme's passes to
# flang's MLIR pipeline (fir::registerPassPipelineConfigCallback) from its
# static initializer. Its MLIR/FIR/flang symbols are left undefined and resolve
# from the host `flang -fc1` at load, so no second copy of LLVM is pulled in.
add_llvm_library(
  FlangEnzymeMLIR-${LLVM_VERSION_MAJOR} MODULE FlangEnzymeMLIRPlugin.cpp
  # FlangEnzymeMLIRLoader.cpp is built into FlangEnzyme instead.
  PARTIAL_SOURCES_INTENDED DISABLE_LLVM_LINK_LLVM_DYLIB)
target_include_directories(
  FlangEnzymeMLIR-${LLVM_VERSION_MAJOR}
  PRIVATE ${FLANG_INCLUDE_DIRS})
target_link_options(
  FlangEnzymeMLIR-${LLVM_VERSION_MAJOR}
  PRIVATE -Wl,--allow-shlib-undefined)
install(
  TARGETS FlangEnzymeMLIR-${LLVM_VERSION_MAJOR}
  LIBRARY DESTINATION lib COMPONENT shlib)

# The LLVM pass plugin for flang loads FlangEnzymeMLIR when flang loads it,
# so that `flang -fpass-plugin=FlangEnzyme-<v>.so` (or FlangEnzyme-<v>.cfg)
# alone brings in Enzyme's MLIR passes too. See FlangEnzymeMLIRLoader.cpp.
if(ENZYME_FLANG_MLIR_LOADER
   AND TARGET FlangEnzyme-${LLVM_VERSION_MAJOR} AND NOT WIN32)
  target_sources(
    FlangEnzyme-${LLVM_VERSION_MAJOR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/FlangEnzymeMLIRLoader.cpp)
  target_compile_definitions(
    FlangEnzyme-${LLVM_VERSION_MAJOR}
    PRIVATE
      FLANG_ENZYME_MLIR_FILE="FlangEnzymeMLIR-${LLVM_VERSION_MAJOR}${CMAKE_SHARED_MODULE_SUFFIX}")
  target_link_libraries(
    FlangEnzyme-${LLVM_VERSION_MAJOR}
    PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(FlangEnzyme-${LLVM_VERSION_MAJOR}
                   FlangEnzymeMLIR-${LLVM_VERSION_MAJOR})
endif()
