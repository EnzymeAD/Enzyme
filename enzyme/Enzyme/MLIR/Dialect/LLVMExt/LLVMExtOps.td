#ifndef ENZYME_DIALECT_LLVMEXT_OPS_TD
#define ENZYME_DIALECT_LLVMEXT_OPS_TD

include "Dialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LLVMPtr : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">>;

// The memory space a pointer's underlying allocation lives in. Space 0 is host
// memory, reachable by plain loads/stores and copyable with memcpy. A non-zero
// space is device memory: it is only addressable by the accelerator, so
// allocating, copying and freeing it all require runtime calls rather than
// their host equivalents. Defaults to 0 so IR written before this attribute
// existed keeps its host meaning.
def MemorySpace : DefaultValuedAttr<I64Attr, "0">;

def AllocOp : LLVMExtOp<"alloc", []> {
  let summary = "Allocates memory";
  let description = [{
    Allocates `size` bytes in `memory_space`. Host allocations (space 0) lower
    to `malloc`, device allocations lower to a `gpu.alloc` in that space.
  }];
  let arguments = (ins I64 : $size, MemorySpace : $memory_space);
  let results = (outs Res<LLVMPtr, "allocated ptr", [MemAlloc]> : $result);

  let assemblyFormat = [{
    $size attr-dict `:` functional-type($size, results)
  }];
}

def FreeOp : LLVMExtOp<"free", []> {
  let summary = "Frees memory";
  let description = [{
    Frees a pointer allocated in `memory_space`, which must agree with the
    `llvm_ext.alloc` that produced it: freeing device memory with the host
    deallocator (or vice versa) corrupts the heap.

    The attribute is optional because a free is usually emitted far from the
    matching alloc -- reverse-mode AD frees a checkpoint buffer it recovered
    from a cache, so the pointer's defining op is an `enzyme.pop` and the space
    isn't apparent locally. When it is omitted, lower-llvm-ext recovers the
    space by tracing the pointer back to its allocation, and fails if it can't:
    guessing "host" would silently free device memory.
  }];
  let arguments = (ins Arg<LLVMPtr, "ptr to free", [MemFree]> : $ptr,
                       OptionalAttr<I64Attr> : $memory_space);
  let results = (outs);

  let assemblyFormat = [{
    $ptr attr-dict `:` type($ptr)
  }];
}

def MemcpyOp : LLVMExtOp<"memcpy", []> {
  let summary = "Copies memory between two pointers in a given memory space";
  let description = [{
    Copies `size` bytes from `src` to `dst`, both of which live in
    `memory_space`. Host-to-host copies (space 0) lower to `llvm.memcpy`;
    copies within a device space lower to a device-to-device runtime copy.
  }];
  let arguments = (ins Arg<LLVMPtr, "destination", [MemWrite]> : $dst,
                       Arg<LLVMPtr, "source", [MemRead]> : $src,
                       I64 : $size,
                       MemorySpace : $memory_space);
  let results = (outs);

  let assemblyFormat = [{
    $dst `,` $src `,` $size attr-dict `:` type($dst) `,` type($src) `,` type($size)
  }];
}

def PtrSizeHintOp : LLVMExtOp<"ptr_size_hint", []> {
  let summary = "Gives information about the underlying memory of a pointer";
  let description = [{
    Describes an allocation Enzyme did not make itself, so that it can be
    cloned (for checkpointing) without having to reason about its provenance:
    `size` is the number of bytes reachable from `ptr`, and `memory_space` says
    whether those bytes are host (0) or device (non-zero) memory. The op has no
    runtime effect and is erased once it has been consumed.
  }];
  let arguments = (ins LLVMPtr : $ptr, AnyInteger : $size,
                       MemorySpace : $memory_space);
  let results = (outs);

  let assemblyFormat = [{
    $ptr `,` $size attr-dict `:` type($ptr) `,` type($size)
  }];
}

#endif // ENZYME_DIALECT_LLVMEXT_OPS_TD
