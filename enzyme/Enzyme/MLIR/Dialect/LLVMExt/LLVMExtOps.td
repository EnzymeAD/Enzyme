#ifndef ENZYME_DIALECT_LLVMEXT_OPS_TD
#define ENZYME_DIALECT_LLVMEXT_OPS_TD

include "Dialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LLVMPtr : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">>;

def AllocOp : LLVMExtOp<"alloc", []> {
  let summary = "Allocates memory";
  let arguments = (ins I64 : $size);
  let results = (outs Res<LLVMPtr, "allocated ptr", [MemAlloc]> : $result);

  let assemblyFormat = [{
    $size attr-dict `:` functional-type($size, results)
  }];
}

def FreeOp : LLVMExtOp<"free", []> {
  let summary = "Frees memory";
  let arguments = (ins Arg<LLVMPtr, "ptr to free", [MemFree]> : $ptr);
  let results = (outs);

  let assemblyFormat = [{
    $ptr attr-dict `:` type($ptr)
  }];
}

def PtrSizeHintOp : LLVMExtOp<"ptr_size_hint", []> {
  let summary = "Gives information about the size of the underlying memory";
  let arguments = (ins LLVMPtr : $ptr, AnyInteger : $size);
  let results = (outs);

  let assemblyFormat = [{
    $ptr `,` $size attr-dict `:` type($ptr) `,` type($size)
  }];
}

#endif // ENZYME_DIALECT_LLVMEXT_OPS_TD
