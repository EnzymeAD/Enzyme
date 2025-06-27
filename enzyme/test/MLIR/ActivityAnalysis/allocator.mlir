// RUN: %eopt --print-activity-analysis='funcs=_Z10reduce_maxPdi annotate' %s 2>&1 | FileCheck %s

module attributes {
    dlti.dl_spec = #dlti.dl_spec<
      #dlti.dl_entry<f16, dense<16> : vector<2xi64>>,
      #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi64>>,
      #dlti.dl_entry<i32, dense<32> : vector<2xi64>>,
      #dlti.dl_entry<f128, dense<128> : vector<2xi64>>,
      #dlti.dl_entry<f64, dense<64> : vector<2xi64>>,
      #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>,
      #dlti.dl_entry<i16, dense<16> : vector<2xi64>>,
      #dlti.dl_entry<i8, dense<8> : vector<2xi64>>,
      #dlti.dl_entry<i1, dense<8> : vector<2xi64>>,
      #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.mlir.global private unnamed_addr constant @".str.1"("d_reduce_max(%i)=%f\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  func.func private @_ZNSt16allocator_traitsISaIdEE8allocateERS0_m(%arg0: i64) -> !llvm.ptr {
    %0 = llvm.mlir.constant(3 : i64) : i64
    %1 = llvm.shl %arg0, %0  : i64
    %2 = func.call @_Znwm(%1) : (i64) -> !llvm.ptr
    %3 = llvm.bitcast %2 : !llvm.ptr to !llvm.ptr
    return %3 : !llvm.ptr
  }
  // CHECK-LABEL: @_Z10reduce_maxPdi:
  // CHECK:         "arg0": Active
  // CHECK:         "allocator": Active
  // CHECK:         "loaded": Active
  llvm.func @_Z10reduce_maxPdi(%arg0: f64 {enzyme.tag = "arg0"}) -> f64 {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = func.call @_ZNSt16allocator_traitsISaIdEE8allocateERS0_m(%0) {tag = "allocator"} : (i64) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 8 : i64} : f64, !llvm.ptr
    %2 = llvm.load %1 {alignment = 8 : i64, tag = "loaded"} : !llvm.ptr -> f64
    llvm.return %2 : f64
  }
  llvm.func @_Z17__enzyme_autodiffPvPdS0_i(...) -> f64
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.addressof @_Z10reduce_maxPdi : !llvm.ptr
    %1 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %2 = llvm.mlir.constant("d_reduce_max(%i)=%f\0A\00") : !llvm.array<21 x i8>
    %3 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.call @_Z17__enzyme_autodiffPvPdS0_i(%0, %1) vararg(!llvm.func<f64 (...)>) : (!llvm.ptr, f64) -> f64
    %6 = llvm.call @printf(%3, %4, %5) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, f64) -> i32
    llvm.return %4 : i32
  }
  llvm.func @printf(!llvm.ptr {llvm.nocapture, llvm.readonly}, ...) -> i32
  func.func private @_Znwm(i64) -> (!llvm.ptr {llvm.noalias, llvm.nonnull})
}
