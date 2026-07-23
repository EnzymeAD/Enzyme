include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# fir-enzyme-opt (see below) links Flang's HLFIR dialect so the Tier-1 hlfir.*
# autodiff models can be built. Resolve Flang up front so the flang-dependent
# HLFIR implementation library in Implementations/ can be defined conditionally.
# FlangConfig references clang targets, so import Clang first. Pass
# -DFlang_DIR=<llvm-build>/lib/cmake/flang -DClang_DIR=<llvm-build>/lib/cmake/clang.
option(ENZYME_FIR_TOOL "Build the fir-enzyme-opt standalone tool (needs Flang)" OFF)
if (ENZYME_FIR_TOOL)
  find_package(Clang REQUIRED CONFIG)
  find_package(Flang REQUIRED CONFIG)
endif()

add_subdirectory(Analysis)
add_subdirectory(Dialect)
add_subdirectory(Interfaces)
add_subdirectory(Implementations)
add_subdirectory(Passes)
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
	MLIRFuncInlinerExtension
        )
add_llvm_executable(enzymemlir-opt enzymemlir-opt.cpp PARTIAL_SOURCES_INTENDED)

install(TARGETS enzymemlir-opt
    EXPORT EnzymeTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT enzymemlir-opt)

llvm_update_compile_flags(enzymemlir-opt)
target_link_libraries(enzymemlir-opt PRIVATE ${LIBS})

# FIREnzyme: Enzyme-MLIR packaged as a dialect+pass plugin loadable into any
# MlirOptMain-based tool (notably Flang's fir-opt), so hlfir.* array intrinsics
# can be differentiated while they are still ops. See enzyme-fir-plugin.cpp and
# PLAN_flang_enzyme_mlir.md. Built with ENZYME_MLIR=ON; loadable only into a
# fir-opt sharing a non-static libMLIR/libLLVM (see FIR_README.md).
option(ENZYME_FIR_PLUGIN "Build the FIREnzyme fir-opt/MLIR plugin" ON)
if (ENZYME_FIR_PLUGIN)
  add_llvm_library(FIREnzyme-${LLVM_VERSION_MAJOR}
    MODULE
    enzyme-fir-plugin.cpp
    PARTIAL_SOURCES_INTENDED
    PLUGIN_TOOL fir-opt
    )
  llvm_update_compile_flags(FIREnzyme-${LLVM_VERSION_MAJOR})
  # Self-contained: the plugin bundles all of Enzyme + the MLIR it needs. At
  # load time fir-opt (built with export_executable_symbols_for_plugins) is in
  # the global scope, so the shared MLIR core interposes to the host's single
  # copy -- one TypeID/dialect registry -- while plugin-only pieces (Enzyme
  # dialects, LinalgTransforms that fir-opt doesn't link) resolve internally.
  # MLIROptLib is deliberately excluded: its MlirOptMain cl::opts would collide
  # with fir-opt's on load.
  target_link_libraries(FIREnzyme-${LLVM_VERSION_MAJOR} PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIREnzymeAnalysis
    MLIREnzymeAutoDiffInterface
    MLIREnzymeImplementations
    MLIREnzymeTransforms
    MLIREnzyme
    MLIRLLVMExt
    MLIRImpulse
    MLIRLinalgTransforms
    MLIRFuncInlinerExtension
    )
  install(TARGETS FIREnzyme-${LLVM_VERSION_MAJOR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT FIREnzyme)
endif()

# fir-enzyme-opt: fir-opt with Enzyme-MLIR statically linked into one binary.
# This is the delivery vehicle for a fully-static LLVM build, where loading a
# second static LLVM copy into fir-opt would double-register cl::opt options.
# Flang/Clang are resolved at the top of this file. See fir-enzyme-opt.cpp and
# PLAN_flang_enzyme_mlir.md.
if (ENZYME_FIR_TOOL)
  add_llvm_executable(fir-enzyme-opt fir-enzyme-opt.cpp PARTIAL_SOURCES_INTENDED)
  llvm_update_compile_flags(fir-enzyme-opt)
  target_include_directories(fir-enzyme-opt PRIVATE ${FLANG_INCLUDE_DIRS})
  target_link_libraries(fir-enzyme-opt PRIVATE
    # FIR/HLFIR, mirroring flang/tools/fir-opt/CMakeLists.txt.
    CUFAttrs
    CUFDialect
    FIRDialect
    FIRSupport
    FIRTransforms
    FIRCodeGen
    FIRCodeGenDialect
    HLFIRDialect
    HLFIRTransforms
    FIROpenACCSupport
    FIROpenACCTransforms
    FIROpenMPSupport
    FlangOpenMPTransforms
    FIRAnalysis
    MIFDialect
    flangPasses
    # Enzyme.
    MLIREnzymeHLFIRImplementations
    MLIREnzymeAnalysis
    MLIREnzymeAutoDiffInterface
    MLIREnzymeImplementations
    MLIREnzymeTransforms
    MLIREnzyme
    MLIRLLVMExt
    MLIRImpulse
    # MLIR.
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIRLinalgTransforms
    MLIRFuncInlinerExtension
    MLIROptLib
    )
  install(TARGETS fir-enzyme-opt
    RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
    COMPONENT fir-enzyme-opt)
endif()
