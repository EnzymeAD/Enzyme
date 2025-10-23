add_mlir_dialect(LLVMExtOps llvm_ext)
add_mlir_doc(LLVMExtDialect LLVMExtDialect Enzyme/ -gen-dialect-doc)
add_mlir_doc(LLVMExtOps LLVMExtOps Enzyme/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS Ops.td)
add_mlir_dialect_library(MLIRLLVMExt
  LLVMExtDialect.cpp
  LLVMExtOps.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  MLIRLLVMExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
)
