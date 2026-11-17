# A plugin for `flang -fc1 -load`. MLIR, FIR and flang symbols stay undefined
# and resolve from the host, so the plugin carries no second copy of LLVM.
add_llvm_library(
  FlangEnzymeMLIR-${LLVM_VERSION_MAJOR}
  MODULE
  FlangEnzymeMLIRPlugin.cpp
  FIRTypeAnnotations.cpp
  # FlangEnzymeMLIRLoader.cpp is built into FlangEnzyme instead.
  PARTIAL_SOURCES_INTENDED
  DISABLE_LLVM_LINK_LLVM_DYLIB)
add_dependencies(FlangEnzymeMLIR-${LLVM_VERSION_MAJOR} MLIREnzyme
                 MLIREnzymeToLLVMIRTranslation)
target_include_directories(
  FlangEnzymeMLIR-${LLVM_VERSION_MAJOR}
  PRIVATE ${FLANG_INCLUDE_DIRS})
# The enzyme dialect and its translation carry the type annotations to LLVM
# IR; bundle them by file, so that their MLIR dependencies resolve from the
# host.
target_link_libraries(
  FlangEnzymeMLIR-${LLVM_VERSION_MAJOR}
  PRIVATE $<TARGET_FILE:MLIREnzymeToLLVMIRTranslation> $<TARGET_FILE:MLIREnzyme>)
target_link_options(
  FlangEnzymeMLIR-${LLVM_VERSION_MAJOR}
  PRIVATE -Wl,--allow-shlib-undefined)
install(
  TARGETS FlangEnzymeMLIR-${LLVM_VERSION_MAJOR}
  LIBRARY DESTINATION lib COMPONENT shlib)

# Make `flang -fpass-plugin=FlangEnzyme-<v>.so` load FlangEnzymeMLIR as well.
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
