set(LLVM_TARGET_DEFINITIONS AffineDerivatives.td)
enzyme_tablegen(AffineDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(AffineDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS ArithDerivatives.td)
enzyme_tablegen(ArithDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(ArithDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS LLVMDerivatives.td)
enzyme_tablegen(LLVMDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(LLVMDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS LLVMExtDerivatives.td)
enzyme_tablegen(LLVMExtDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(LLVMExtDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS LinalgDerivatives.td)
enzyme_tablegen(LinalgDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(LinalgDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS NVVMDerivatives.td)
enzyme_tablegen(NVVMDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(NVVMDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS SCFDerivatives.td)
enzyme_tablegen(SCFDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(SCFDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS CFDerivatives.td)
enzyme_tablegen(CFDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(CFDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS MemRefDerivatives.td)
enzyme_tablegen(MemRefDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(MemRefDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS ComplexDerivatives.td)
enzyme_tablegen(ComplexDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(ComplexDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS MathDerivatives.td)
enzyme_tablegen(MathDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(MathDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS FuncDerivatives.td)
enzyme_tablegen(FuncDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(FuncDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS EnzymeDerivatives.td)
enzyme_tablegen(EnzymeDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(EnzymeDerivativesIncGen)

add_mlir_library(
  MLIREnzymeImplementations
  AffineAutoDiffOpInterfaceImpl.cpp
  ArithAutoDiffOpInterfaceImpl.cpp
  CoreDialectsAutoDiffImplementations.cpp
  CoreDialectsAutoDiffRegistration.cpp
  LLVMAutoDiffOpInterfaceImpl.cpp
  LLVMExtAutoDiffOpInterfaceImpl.cpp
  NVVMAutoDiffOpInterfaceImpl.cpp
  MemRefAutoDiffOpInterfaceImpl.cpp
  ComplexAutoDiffOpInterfaceImpl.cpp
  FuncAutoDiffOpInterfaceImpl.cpp
  LinalgAutoDiffOpInterfaceImpl.cpp
  BuiltinAutoDiffTypeInterfaceImpl.cpp
  SCFAutoDiffOpInterfaceImpl.cpp
  CFAutoDiffOpInterfaceImpl.cpp
  MathAutoDiffOpInterfaceImpl.cpp
  GPUAutoDiffOpInterfaceImpl.cpp
  EnzymeAutoDiffOpInterfaceImpl.cpp
  DEPENDS
  MLIRAutoDiffOpInterfaceIncGen
  AffineDerivativesIncGen
  ArithDerivativesIncGen
  ComplexDerivativesIncGen
  LLVMDerivativesIncGen
  LLVMExtDerivativesIncGen
  LinalgDerivativesIncGen
  FuncDerivativesIncGen
  NVVMDerivativesIncGen
  SCFDerivativesIncGen
  CFDerivativesIncGen
  MemRefDerivativesIncGen
  MathDerivativesIncGen
  EnzymeDerivativesIncGen
  MLIREnzymeEnumsIncGen
  LINK_LIBS
  PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRComplexDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRGPUDialect
  MLIREnzymeAutoDiffInterface
  MLIRIR
  MLIRSCFDialect
  MLIRLinalgDialect)

# FlangEnzymeMLIR: a plugin for `flang -fc1 -load` that adds Enzyme's passes to
# flang's MLIR pipeline (fir::registerPassPipelineConfigCallback) from its
# static initializer. Its MLIR/FIR/flang symbols are left undefined and resolve
# from the host `flang -fc1` at load, so no second copy of LLVM is pulled in.
if(ENZYME_FLANG_MLIR)
  add_llvm_library(
    FlangEnzymeMLIR-${LLVM_VERSION_MAJOR} MODULE
    FlangEnzymeMLIRPlugin.cpp PARTIAL_SOURCES_INTENDED
    DISABLE_LLVM_LINK_LLVM_DYLIB)
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
  if(TARGET FlangEnzyme-${LLVM_VERSION_MAJOR} AND NOT WIN32)
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
endif()
