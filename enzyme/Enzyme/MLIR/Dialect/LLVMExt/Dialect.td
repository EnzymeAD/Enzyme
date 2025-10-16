#ifndef ENZYME_AD_JAX_DIALECT_LLVMEXT_DIALECT_TD
#define ENZYME_AD_JAX_DIALECT_LLVMEXT_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LLVMExt dialect definition.
//===----------------------------------------------------------------------===//

def LLVMExtDialect : Dialect {
  let name = "llvm_ext";
  let summary = "A dialect for extending LLVM";
  let cppNamespace = "::mlir::enzyme::llvm_ext";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Base LLVMExt operation definition.
//===----------------------------------------------------------------------===//

class LLVMExtOp<string mnemonic, list<Trait> traits = []>
    : Op<LLVMExtDialect, mnemonic, traits>;

class LLVMExtType<string name> : TypeDef<LLVMExtDialect, name>;

#endif // ENZYME_AD_JAX_DIALECT_LLVMEXT_DIALECT_TD
