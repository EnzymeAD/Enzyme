set(LLVM_TARGET_DEFINITIONS AffineDerivatives.td)
enzyme_tablegen(AffineDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(AffineDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS ArithDerivatives.td)
enzyme_tablegen(ArithDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(ArithDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS LLVMDerivatives.td)
enzyme_tablegen(LLVMDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(LLVMDerivativesIncGen)

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

add_mlir_library(MLIREnzymeImplementations
  AffineAutoDiffOpInterfaceImpl.cpp
  ArithAutoDiffOpInterfaceImpl.cpp
  CoreDialectsAutoDiffImplementations.cpp
  LLVMAutoDiffOpInterfaceImpl.cpp
  NVVMAutoDiffOpInterfaceImpl.cpp
  MemRefAutoDiffOpInterfaceImpl.cpp
  LinalgAutoDiffOpInterfaceImpl.cpp
  BuiltinAutoDiffTypeInterfaceImpl.cpp
  SCFAutoDiffOpInterfaceImpl.cpp
  CFAutoDiffOpInterfaceImpl.cpp

  DEPENDS
  MLIRAutoDiffOpInterfaceIncGen
  AffineDerivativesIncGen
  ArithDerivativesIncGen
  LLVMDerivativesIncGen
  NVVMDerivativesIncGen
  SCFDerivativesIncGen
  CFDerivativesIncGen
  MemRefDerivativesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIREnzymeAutoDiffInterface
  MLIRIR
  MLIRSCFDialect
  MLIRLinalgDialect
)
