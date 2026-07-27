include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Resolve Flang up front, so Implementations/ can define the flang-dependent
# library conditionally. FlangConfig references clang targets, so import Clang
# first (pass -DFlang_DIR=<llvm-build>/lib/cmake/flang and likewise -DClang_DIR).
option(ENZYME_FLANG_MLIR "Build the Flang-dependent FIR/HLFIR autodiff layer (needs Flang)" OFF)
if (ENZYME_FLANG_MLIR)
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

# FIREnzyme: Enzyme-MLIR as a dialect+pass plugin for any MlirOptMain-based tool,
# notably Flang's fir-opt. See enzyme-fir-plugin.cpp. It bundles the
# flang-dependent HLFIR layer, which only exists once ENZYME_FLANG_MLIR is on and
# Flang was found.
option(ENZYME_FIR_PLUGIN "Build the FIREnzyme fir-opt/MLIR plugin" ON)
if (ENZYME_FIR_PLUGIN AND TARGET MLIREnzymeHLFIRImplementations)
  add_llvm_library(FIREnzyme-${LLVM_VERSION_MAJOR}
    MODULE
    enzyme-fir-plugin.cpp
    PARTIAL_SOURCES_INTENDED
    DISABLE_LLVM_LINK_LLVM_DYLIB
    )
  llvm_update_compile_flags(FIREnzyme-${LLVM_VERSION_MAJOR})
  add_dependencies(FIREnzyme-${LLVM_VERSION_MAJOR}
    MLIREnzymeHLFIRImplementations
    MLIREnzymeImplementations
    MLIREnzymeTransforms
    MLIREnzymeAnalysis
    MLIRLLVMExt
    MLIRImpulse
    MLIREnzyme
    MLIREnzymeAutoDiffInterface)
  # Bundle only the Enzyme archives (plus the upstream dialects fir-opt does not
  # carry, see below) and leave every other MLIR/FIR/LLVM symbol undefined, to be
  # resolved from the host fir-opt at load. Bundling ${dialect_libs} instead drags
  # in a second copy of the llvm::cl::opt globals and aborts with "registered more
  # than once".
  target_link_libraries(FIREnzyme-${LLVM_VERSION_MAJOR} PRIVATE
    -Wl,--start-group
    # The flang-dependent HLFIR layer; its FIR/HLFIR/flang symbols stay
    # undefined and resolve from the host.
    $<TARGET_FILE:MLIREnzymeHLFIRImplementations>
    $<TARGET_FILE:MLIREnzymeImplementations>
    $<TARGET_FILE:MLIREnzymeTransforms>
    $<TARGET_FILE:MLIREnzymeAnalysis>
    # Enzyme's own dialects (referenced by the models/passes above).
    $<TARGET_FILE:MLIRLLVMExt>
    $<TARGET_FILE:MLIRImpulse>
    $<TARGET_FILE:MLIREnzyme>
    $<TARGET_FILE:MLIREnzymeAutoDiffInterface>
    # Two dialects fir-opt does not register: Tensor, needed by the tensor
    # autodiff model, and Linalg, because Transforms/Utils.cpp builds
    # linalg.generic ops whatever models are registered. Bundled by file, so
    # their own MLIR deps resolve from the host. (fir-opt does provide affine.)
    $<TARGET_FILE:MLIRTensorDialect>
    $<TARGET_FILE:MLIRLinalgDialect>
    $<TARGET_FILE:MLIRLinalgTransforms>
    -Wl,--end-group)
  target_link_options(FIREnzyme-${LLVM_VERSION_MAJOR} PRIVATE
    -Wl,--allow-shlib-undefined)
  install(TARGETS FIREnzyme-${LLVM_VERSION_MAJOR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT FIREnzyme)
endif()
