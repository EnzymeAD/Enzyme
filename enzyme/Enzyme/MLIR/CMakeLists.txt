include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# The flang-dependent FIR/HLFIR autodiff layer (Implementations/, and the
# FIREnzyme fir-opt plugin below) links Flang's HLFIR dialect so the Tier-1
# hlfir.* autodiff models can be built. Resolve Flang up front so the
# flang-dependent HLFIR implementation library in Implementations/ can be defined
# conditionally. FlangConfig references clang targets, so import Clang first. Pass
# -DFlang_DIR=<llvm-build>/lib/cmake/flang -DClang_DIR=<llvm-build>/lib/cmake/clang.
option(ENZYME_FIR_TOOL "Build the Flang-dependent FIR/HLFIR autodiff layer (needs Flang)" OFF)
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
# PLAN_flang_enzyme_mlir.md. Built with ENZYME_MLIR=ON.
option(ENZYME_FIR_PLUGIN "Build the FIREnzyme fir-opt/MLIR plugin" ON)
# The plugin bundles the flang-dependent HLFIR layer (shared Fortran
# registration + the enzyme-lower-fortran-calls pass), which only exists when
# the FIR layer is configured (ENZYME_FIR_TOOL -> Flang found).
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
  # Lean, exactly like the FlangEnzymeMLIR flang plugin: bundle *only* the Enzyme
  # static archives (plus the two upstream dialects fir-opt does not itself carry,
  # see below), and leave all other MLIR/FIR/LLVM symbols undefined so they
  # resolve from the host fir-opt at load. fir-opt is built with
  # export_executable_symbols_for_plugins, so its single statically-linked copy
  # of MLIR/LLVM satisfies them -- crucially including the llvm::cl::opt globals,
  # so there is exactly one copy of each command-line option and no "registered
  # more than once" clash on dlopen. This is what lets the plugin load into a
  # *statically* linked fir-opt; bundling ${dialect_libs}/${conversion_libs}
  # instead would drag in a second copy of LLVM's cl::opts and abort. See
  # FIR_README.md.
  target_link_libraries(FIREnzyme-${LLVM_VERSION_MAJOR} PRIVATE
    -Wl,--start-group
    # The flang-dependent HLFIR layer: shared Fortran registration
    # (registerEnzymeFortranInterfaces) + the enzyme-lower-fortran-calls pass.
    # Its FIR/HLFIR/flang symbols stay undefined and resolve from the host.
    $<TARGET_FILE:MLIREnzymeHLFIRImplementations>
    $<TARGET_FILE:MLIREnzymeImplementations>
    $<TARGET_FILE:MLIREnzymeTransforms>
    $<TARGET_FILE:MLIREnzymeAnalysis>
    # Enzyme's own dialects (referenced by the models/passes above).
    $<TARGET_FILE:MLIRLLVMExt>
    $<TARGET_FILE:MLIRImpulse>
    $<TARGET_FILE:MLIREnzyme>
    $<TARGET_FILE:MLIREnzymeAutoDiffInterface>
    # Two upstream dialects fir-opt does NOT register (unlike flang): Tensor and
    # Linalg. Tensor is needed by the tensor autodiff model; Linalg because
    # Enzyme's own transform utilities (Transforms/Utils.cpp) build linalg.generic
    # ops, so MLIREnzymeTransforms references the Linalg dialect + its transforms
    # regardless of which autodiff models are registered. Bundle these by file so
    # their own MLIR deps stay undefined and resolve from the host; where the host
    # defines a symbol it preempts the bundled copy. (fir-opt does provide affine,
    # so no Affine archive is needed.)
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
