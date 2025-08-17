set(LLVM_TARGET_DEFINITIONS AffineDerivatives.td)
enzyme_tablegen(AffineDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(AffineDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS ArithDerivatives.td)
enzyme_tablegen(ArithDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(ArithDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS LLVMDerivatives.td)
enzyme_tablegen(LLVMDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(LLVMDerivativesIncGen)

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

add_mlir_library(MLIREnzymeImplementations
  AffineAutoDiffOpInterfaceImpl.cpp
  ArithAutoDiffOpInterfaceImpl.cpp
  CoreDialectsAutoDiffImplementations.cpp
  LLVMAutoDiffOpInterfaceImpl.cpp
  NVVMAutoDiffOpInterfaceImpl.cpp
  MemRefAutoDiffOpInterfaceImpl.cpp
  ComplexAutoDiffOpInterfaceImpl.cpp
  FuncAutoDiffOpInterfaceImpl.cpp
  LinalgAutoDiffOpInterfaceImpl.cpp
  BuiltinAutoDiffTypeInterfaceImpl.cpp
  SCFAutoDiffOpInterfaceImpl.cpp
  CFAutoDiffOpInterfaceImpl.cpp
  MathAutoDiffOpInterfaceImpl.cpp
  EnzymeAutoDiffOpInterfaceImpl.cpp

  DEPENDS
  MLIRAutoDiffOpInterfaceIncGen
  AffineDerivativesIncGen
  ArithDerivativesIncGen
  ComplexDerivativesIncGen
  LLVMDerivativesIncGen
  LinalgDerivativesIncGen
  FuncDerivativesIncGen
  NVVMDerivativesIncGen
  SCFDerivativesIncGen
  CFDerivativesIncGen
  MemRefDerivativesIncGen
  MathDerivativesIncGen
  EnzymeDerivativesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRComplexDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIREnzymeAutoDiffInterface
  MLIRIR
  MLIRSCFDialect
  MLIRLinalgDialect
)
