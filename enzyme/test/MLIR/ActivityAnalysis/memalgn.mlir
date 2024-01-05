// RUN: %eopt --print-activity-analysis=funcs=memalgn %s 2>&1 | FileCheck %s

module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<i1, dense<8> : vector<2xi64>>,
    #dlti.dl_entry<i8, dense<8> : vector<2xi64>>,
    #dlti.dl_entry<i16, dense<16> : vector<2xi64>>,
    #dlti.dl_entry<i32, dense<32> : vector<2xi64>>,
    #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi64>>,
    #dlti.dl_entry<f16, dense<16> : vector<2xi64>>,
    #dlti.dl_entry<f32, dense<64> : vector<2xi64>>,
    #dlti.dl_entry<f128, dense<128> : vector<2xi64>>>} {
  // Test aliasing of dense analysis (%5, which is stored to, aliases with %6, which is loaded from)
  // CHECK-LABEL: @memalgn:
  // CHECK:         "arg0": Active
  // CHECK:         "retval": Active
  llvm.func @memalgn(%arg0: f32 {enzyme.tag = "arg0"}) -> f32 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(8 : i64) : i64
    %2 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.bitcast %2 : !llvm.ptr to !llvm.ptr
    %4 = llvm.call @posix_memalign(%3, %1, %1) : (!llvm.ptr, i64, i64) -> i32
    %5 = llvm.load %2 : !llvm.ptr -> !llvm.ptr
    llvm.store %arg0, %5 : f32, !llvm.ptr
    %6 = llvm.load %2 : !llvm.ptr -> !llvm.ptr
    %7 = llvm.load %6 {tag = "retval"} : !llvm.ptr -> f32
    llvm.return %7 : f32
  }
  llvm.func @posix_memalign(!llvm.ptr, i64, i64) -> i32
}
