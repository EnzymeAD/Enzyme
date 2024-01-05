// RUN: %eopt --test-print-alias-analysis --split-input-file %s 2>&1 | FileCheck %s

module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<i1, dense<8> : vector<2xi64>>,
    #dlti.dl_entry<i8, dense<8> : vector<2xi64>>,
    #dlti.dl_entry<i16, dense<16> : vector<2xi64>>,
    #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>,
    #dlti.dl_entry<f64, dense<64> : vector<2xi64>>,
    #dlti.dl_entry<f128, dense<128> : vector<2xi64>>,
    #dlti.dl_entry<i32, dense<32> : vector<2xi64>>,
    #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi64>>,
    #dlti.dl_entry<f16, dense<16> : vector<2xi64>>,
    #dlti.dl_entry<"dlti.endianness", "little">>} {
  // CHECK: "store dest" and "load source": MayAlias
  llvm.func internal @f(%arg0: f64) -> f64 attributes {dso_local} {
    %0 = llvm.mlir.constant(10 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(dense<0> : vector<2xi64>) : vector<2xi64>
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(5 : i64) : i64
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.alloca %0 x i8 {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %7 = llvm.ptrtoint %6 : !llvm.ptr to i64
    %8 = llvm.insertelement %7, %2[%3 : i32] : vector<2xi64>
    %9 = llvm.insertelement %4, %8[%5 : i32] : vector<2xi64>
    %10 = llvm.bitcast %6 {tag = "store dest"} : !llvm.ptr to !llvm.ptr
    llvm.store %arg0, %10 {alignment = 8 : i64} : f64, !llvm.ptr
    %11 = llvm.bitcast %9 : vector<2xi64> to i128
    %12 = llvm.trunc %11 : i128 to i64
    %13 = llvm.inttoptr %12 {tag = "load source"} : i64 to !llvm.ptr
    %14 = llvm.load %13 {alignment = 8 : i64} : !llvm.ptr -> f64
    llvm.return %14 : f64
  }
}
