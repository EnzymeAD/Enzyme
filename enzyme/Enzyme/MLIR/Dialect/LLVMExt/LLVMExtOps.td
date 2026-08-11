#ifndef ENZYME_DIALECT_LLVMEXT_OPS_TD
#define ENZYME_DIALECT_LLVMEXT_OPS_TD

include "Dialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LLVMPtr : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">>;

// The memory space an allocation lives in is the address space of the
// `!llvm.ptr` that names it: space 0 is host memory, reachable by plain
// loads/stores and copyable with memcpy, while a non-zero space is device
// memory, only addressable by the accelerator, so allocating and freeing it
// require runtime calls rather than their host equivalents. The ops below
// therefore don't carry a memory space of their own -- it would be a second
// source of truth, free to disagree with the types.

def AllocOp : LLVMExtOp<"alloc", []> {
  let summary = "Allocates memory";
  let description = [{
    Allocates `size` bytes in the address space of the result pointer. Host
    allocations (space 0) lower to `malloc`, device allocations lower to a
    `gpu.alloc` in that space.
  }];
  let arguments = (ins I64 : $size);
  let results = (outs Res<LLVMPtr, "allocated ptr", [MemAlloc]> : $result);

  let assemblyFormat = [{
    $size attr-dict `:` functional-type($size, results)
  }];

  let extraClassDeclaration = [{
    /// Memory space this allocation lives in.
    unsigned getMemorySpace() {
      return ::llvm::cast<::mlir::LLVM::LLVMPointerType>(getResult().getType())
          .getAddressSpace();
    }
  }];
}

def FreeOp : LLVMExtOp<"free", []> {
  let summary = "Frees memory";
  let description = [{
    Frees a pointer with the deallocator for the address space of its type:
    freeing device memory with the host deallocator (or vice versa) would
    corrupt the heap.
  }];
  let arguments = (ins Arg<LLVMPtr, "ptr to free", [MemFree]> : $ptr);
  let results = (outs);

  let assemblyFormat = [{
    $ptr attr-dict `:` type($ptr)
  }];

  let extraClassDeclaration = [{
    /// Memory space the freed pointer lives in.
    unsigned getMemorySpace() {
      return ::llvm::cast<::mlir::LLVM::LLVMPointerType>(getPtr().getType())
          .getAddressSpace();
    }
  }];
}

def MemcpyOp : LLVMExtOp<"memcpy", []> {
  let summary = "Copies memory between two pointers in a given memory space";
  let description = [{
    Copies `size` bytes from `src` to `dst`, which must live in the same
    address space. Lowers to an `enzymexla.memcpy` of that space, host memory
    included.
  }];
  let arguments = (ins Arg<LLVMPtr, "destination", [MemWrite]> : $dst,
                       Arg<LLVMPtr, "source", [MemRead]> : $src,
                       I64 : $size);
  let results = (outs);

  let assemblyFormat = [{
    $dst `,` $src `,` $size attr-dict `:` type($dst) `,` type($src) `,` type($size)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Memory space both pointers live in (the verifier keeps them equal).
    unsigned getMemorySpace() {
      return ::llvm::cast<::mlir::LLVM::LLVMPointerType>(getDst().getType())
          .getAddressSpace();
    }
  }];
}

def PtrSizeHintOp : LLVMExtOp<"ptr_size_hint", []> {
  let summary = "Gives information about the underlying memory of a pointer";
  let description = [{
    Describes an allocation Enzyme did not make itself, so that it can be
    cloned (for checkpointing) without having to reason about its provenance:
    `size` is the number of bytes reachable from `ptr`. Whether those bytes are
    host or device memory is the address space of `ptr` itself. The op has no
    runtime effect and is erased once it has been consumed.
  }];
  let arguments = (ins LLVMPtr : $ptr, AnyInteger : $size);
  let results = (outs);

  let assemblyFormat = [{
    $ptr `,` $size attr-dict `:` type($ptr) `,` type($size)
  }];

  let extraClassDeclaration = [{
    /// Memory space the described allocation lives in.
    unsigned getMemorySpace() {
      return ::llvm::cast<::mlir::LLVM::LLVMPointerType>(getPtr().getType())
          .getAddressSpace();
    }
  }];
}

#endif // ENZYME_DIALECT_LLVMEXT_OPS_TD
