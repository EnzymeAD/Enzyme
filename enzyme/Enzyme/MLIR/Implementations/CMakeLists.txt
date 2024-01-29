
set(LLVM_TARGET_DEFINITIONS ArithDerivatives.td)
enzyme_tablegen(ArithDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(ArithDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS LLVMDerivatives.td)
enzyme_tablegen(LLVMDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(LLVMDerivativesIncGen)

add_mlir_library(MLIREnzymeImplementations
  ArithAutoDiffOpInterfaceImpl.cpp
  LLVMAutoDiffOpInterfaceImpl.cpp
  MemRefAutoDiffOpInterfaceImpl.cpp
  LinalgAutoDiffOpInterfaceImpl.cpp
  BuiltinAutoDiffTypeInterfaceImpl.cpp
  SCFAutoDiffOpInterfaceImpl.cpp

  DEPENDS
  MLIRAutoDiffOpInterfaceIncGen
  ArithDerivativesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIREnzymeAutoDiffInterface
  MLIRIR
  MLIRSCFDialect
  MLIRLinalgDialect
)
