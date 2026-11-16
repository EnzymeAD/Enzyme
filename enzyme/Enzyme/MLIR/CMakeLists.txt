include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Resolve Flang up front, so Flang/ can define the flang plugin.
# FlangConfig references clang targets, so import Clang first.
# Pass -DFlang_DIR=<llvm>/lib/cmake/flang and likewise -DClang_DIR.
option(
  ENZYME_FLANG_MLIR
  "Build FlangEnzymeMLIR, Enzyme's plugin for flang's MLIR pipeline (needs Flang)"
  OFF)
include(CMakeDependentOption)
cmake_dependent_option(
  ENZYME_FLANG_MLIR_LOADER
  "Make FlangEnzyme load FlangEnzymeMLIR when flang loads it" ON
  "ENZYME_FLANG_MLIR" OFF)
if(ENZYME_FLANG_MLIR)
  find_package(Clang REQUIRED CONFIG)
  find_package(Flang REQUIRED CONFIG)
endif()
add_subdirectory(Analysis)
add_subdirectory(Dialect)
add_subdirectory(Interfaces)
add_subdirectory(Implementations)
if(ENZYME_FLANG_MLIR)
  add_subdirectory(Flang)
endif()
add_subdirectory(Passes)
add_subdirectory(Target)
add_subdirectory(Integrations)
add_subdirectory(enzymemlir-translate)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
    z
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIREnzymeAnalysis
    MLIREnzymeImplementations
    MLIREnzymeTransforms
    MLIREnzyme
    MLIROptLib
    MLIRFuncInlinerExtension)
add_llvm_executable(enzymemlir-opt enzymemlir-opt.cpp)

install(
  TARGETS enzymemlir-opt
  EXPORT EnzymeTargets
  RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR} COMPONENT enzymemlir-opt)

llvm_update_compile_flags(enzymemlir-opt)
target_link_libraries(enzymemlir-opt PRIVATE ${LIBS})
