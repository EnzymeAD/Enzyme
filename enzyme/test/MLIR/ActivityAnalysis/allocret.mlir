// RUN: %eopt --print-activity-analysis --split-input-file %s 2>&1 | FileCheck %s

module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi64>>,
    #dlti.dl_entry<f16, dense<16> : vector<2xi64>>,
    #dlti.dl_entry<i32, dense<32> : vector<2xi64>>,
    #dlti.dl_entry<f128, dense<128> : vector<2xi64>>,
    #dlti.dl_entry<f64, dense<64> : vector<2xi64>>,
    #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>,
    #dlti.dl_entry<i8, dense<8> : vector<2xi64>>,
    #dlti.dl_entry<i16, dense<16> : vector<2xi64>>,
    #dlti.dl_entry<i1, dense<8> : vector<2xi64>>,
    #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.func @malloc(i64) -> !llvm.ptr
  // CHECK-LABEL: @kernel_main
  // CHECK:         "malloc": Active
  llvm.func @kernel_main(%arg0: f32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.call @malloc(%0) {tag = "malloc"} : (i64) -> !llvm.ptr
    %2 = llvm.bitcast %1 : !llvm.ptr to !llvm.ptr
    llvm.store %arg0, %2 {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.return %2 : !llvm.ptr
  }
}
