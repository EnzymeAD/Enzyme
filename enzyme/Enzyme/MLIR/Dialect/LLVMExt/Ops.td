#ifndef ENZYME_DIALECT_LLVMEXT_OPS_TD
#define ENZYME_DIALECT_LLVMEXT_OPS_TD

include "Dialect.td"

def LLVMPtr : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">>;

def AllocOp : LLVMExtOp<"alloc", []> {
  let summary = "allocate memory";
  let arguments = (ins AnyInteger : $size);
  let results = (outs LLVMPtr : $result);

  let assemblyFormat = [{
    $size attr-dict `:` functional-type($size, results)
  }];
}

def FreeOp : LLVMExtOp<"free", []> {
  let summary = "free memory";
  let arguments = (ins LLVMPtr : $ptr);
  let results = (outs);

  let assemblyFormat = [{
    $ptr attr-dict `:` type($ptr)
  }];
}

def PtrSizeHintOp : LLVMExtOp<"ptr_size_hint", []> {
  let summary = "give information about the size of the underlying memory";
  let arguments = (ins LLVMPtr : $ptr, AnyInteger : $size);
  let results = (outs LLVMPtr : $result);

  let assemblyFormat = [{
    $ptr `,` $size attr-dict `:` functional-type(operands, results)
  }];
}

#endif // ENZYME_DIALECT_LLVMEXT_OPS_TD
