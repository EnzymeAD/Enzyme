// RUN: %eopt --print-activity-analysis='annotate' --split-input-file %s 2>&1 | FileCheck %s

module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<i16, dense<16> : vector<2xi64>>,
    #dlti.dl_entry<i32, dense<32> : vector<2xi64>>,
    #dlti.dl_entry<i8, dense<8> : vector<2xi64>>,
    #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi64>>,
    #dlti.dl_entry<i1, dense<8> : vector<2xi64>>,
    #dlti.dl_entry<f16, dense<16> : vector<2xi64>>,
    #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>,
    #dlti.dl_entry<f64, dense<64> : vector<2xi64>>,
    #dlti.dl_entry<f128, dense<128> : vector<2xi64>>,
    #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.func @f(%arg0: !llvm.ptr {llvm.nocapture, llvm.readonly}) -> !llvm.ptr attributes {memory = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>, sym_visibility = "private"} {
    %0 = llvm.load %arg0 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  // CHECK-LABEL: @submalloced
  // CHECK:         "ptrtoptr": Active
  // CHECK:         "retval": Active
  llvm.func @submalloced(%arg0: !llvm.ptr) -> f64 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64, tag = "ptrtoptr"} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    %2 = llvm.call @f(%1) : (!llvm.ptr) -> !llvm.ptr
    %3 = llvm.load %2 {alignment = 8 : i64, tag = "retval"} : !llvm.ptr -> f64
    llvm.return %3 : f64
  }
}
