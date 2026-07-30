// RUN: not %eopt %s --lower-llvm-ext 2>&1 | FileCheck %s

// Device-space allocations lower through gpu.alloc/gpu.dealloc plus the
// enzymexla view casts that bridge !llvm.ptr and memref. That dialect lives
// outside this repository, so eopt cannot lower these ops -- check that the
// failure is reported rather than silently leaving the ops in place, since an
// unlowered device alloc would otherwise surface much later as a host access to
// device memory.

module {
  llvm.func @clone_device(%src: !llvm.ptr, %n: i64) {
    %0 = llvm_ext.alloc %n {memory_space = 1 : i64} : (i64) -> !llvm.ptr
    llvm_ext.memcpy %0, %src, %n {memory_space = 1 : i64} : !llvm.ptr, !llvm.ptr, i64
    llvm_ext.free %0 {memory_space = 1 : i64} : !llvm.ptr
    llvm.return
  }
}

// CHECK: error: lowering device memory requires the 'enzymexla' dialect to be loaded
