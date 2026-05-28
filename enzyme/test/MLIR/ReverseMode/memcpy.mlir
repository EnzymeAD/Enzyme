// RUN: %eopt --enzyme %s | FileCheck %s

func.func @copy1(%dst: !llvm.ptr, %src: !llvm.ptr, %n: i64) {
  "llvm.intr.memcpy"(%dst, %src, %n)
      <{arg_attrs = [{llvm.align = 8 : i64}], isVolatile = false}>
      : (!llvm.ptr, !llvm.ptr, i64) -> ()
  return
}

func.func @dcopy1(%dst: !llvm.ptr, %ddst: !llvm.ptr,
                  %src: !llvm.ptr, %dsrc: !llvm.ptr, %n: i64) {
  enzyme.autodiff @copy1(%dst, %ddst, %src, %dsrc, %n) {
    activity = [#enzyme<activity enzyme_dup>,
                #enzyme<activity enzyme_dup>,
                #enzyme<activity enzyme_const>],
    ret_activity = []
  } : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
  return
}

func.func @copy2(%dst: !llvm.ptr, %src: !llvm.ptr, %n: i64) {
  "llvm.intr.memcpy"(%dst, %src, %n)
      <{arg_attrs = [{llvm.align = 8 : i64}, {llvm.align = 8 : i64}, {}],
        isVolatile = false}>
      : (!llvm.ptr, !llvm.ptr, i64) -> ()
  return
}

func.func @dcopy2(%dst: !llvm.ptr, %ddst: !llvm.ptr,
                  %src: !llvm.ptr, %dsrc: !llvm.ptr, %n: i64) {
  enzyme.autodiff @copy2(%dst, %ddst, %src, %dsrc, %n) {
    activity = [#enzyme<activity enzyme_dup>,
                #enzyme<activity enzyme_dup>,
                #enzyme<activity enzyme_const>],
    ret_activity = []
  } : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
  return
}

// CHECK-LABEL:   func.func private @diffecopy1(
// Forward: the primal memcpy is preserved.
// CHECK:           "llvm.intr.memcpy"
// Reverse: n / sizeof(f64) element-wise loop, d_src[i] += d_dst[i]; d_dst[i]=0.
// CHECK:           %[[BYTES:.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK:           llvm.sdiv %{{.+}}, %[[BYTES]] : i64
// CHECK:           arith.index_cast
// CHECK:           %[[ZERO:.+]] = arith.constant 0.000000e+00 : f64
// CHECK:           scf.for
// CHECK:             llvm.getelementptr %{{.+}}[%{{.+}}] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK:             llvm.load %{{.+}} : !llvm.ptr -> f64
// CHECK:             llvm.getelementptr %{{.+}}[%{{.+}}] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK:             llvm.load %{{.+}} : !llvm.ptr -> f64
// CHECK:             %[[SUM:.+]] = arith.addf
// CHECK:             llvm.store %[[SUM]], %{{.+}} : f64, !llvm.ptr
// CHECK:             llvm.store %[[ZERO]], %{{.+}} : f64, !llvm.ptr

// CHECK-LABEL:   func.func private @diffecopy2(
// CHECK:           "llvm.intr.memcpy"
// CHECK:           llvm.mlir.constant(8 : i64) : i64
// CHECK:           %[[ZERO2:.+]] = arith.constant 0.000000e+00 : f64
// CHECK:           scf.for
// CHECK:             %[[SUM2:.+]] = arith.addf
// CHECK:             llvm.store %[[SUM2]], %{{.+}} : f64, !llvm.ptr
// CHECK:             llvm.store %[[ZERO2]], %{{.+}} : f64, !llvm.ptr
