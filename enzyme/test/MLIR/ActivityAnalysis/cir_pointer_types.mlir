// RUN: %eopt --print-activity-analysis='dataflow=false annotate=true' %s 2>&1 | FileCheck %s

// Flattened output of clang -fclangir -emit-cir -O0 for
//   double idx(double *a, int *n) { return a[n[1]] * 2.0; }

// CHECK-LABEL: cir.func @idx
// CHECK:         %[[N:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i> {enzyme.ici = true, enzyme.res_icv0 = true}
// CHECK:         %[[NI:.*]] = cir.ptr_stride %[[N]], %{{.*}} -> !cir.ptr<!s32i> {enzyme.ici = true, enzyme.res_icv0 = true}
// CHECK:         cir.load align(4) %[[NI]] : !cir.ptr<!s32i>, !s32i {enzyme.ici = true, enzyme.res_icv0 = true}
// CHECK:         %[[A:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!cir.double>>, !cir.ptr<!cir.double> {enzyme.ici = true, enzyme.res_icv0 = false}
// CHECK:         %[[AI:.*]] = cir.ptr_stride %[[A]], %{{.*}} -> !cir.ptr<!cir.double> {enzyme.ici = true, enzyme.res_icv0 = false}
// CHECK:         cir.load align(8) %[[AI]] : !cir.ptr<!cir.double>, !cir.double {enzyme.ici = false, enzyme.res_icv0 = false}

!s32i = !cir.int<s, 32>
!s64i = !cir.int<s, 64>
module {
  cir.func @idx(%arg0: !cir.ptr<!cir.double>, %arg1: !cir.ptr<!s32i>) -> !cir.double {
    %0 = cir.const #cir.fp<2.000000e+00> : !cir.double
    %1 = cir.const #cir.int<1> : !s64i
    %2 = cir.alloca "a" align(8) init : !cir.ptr<!cir.ptr<!cir.double>>
    %3 = cir.alloca "n" align(8) init : !cir.ptr<!cir.ptr<!s32i>>
    %4 = cir.alloca "__retval" align(8) : !cir.ptr<!cir.double>
    cir.store %arg0, %2 : !cir.ptr<!cir.double>, !cir.ptr<!cir.ptr<!cir.double>>
    cir.store %arg1, %3 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
    %5 = cir.load align(8) %3 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
    %6 = cir.ptr_stride %5, %1 : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
    %7 = cir.load align(4) %6 : !cir.ptr<!s32i>, !s32i
    %8 = cir.cast integral %7 : !s32i -> !s64i
    %9 = cir.load align(8) %2 : !cir.ptr<!cir.ptr<!cir.double>>, !cir.ptr<!cir.double>
    %10 = cir.ptr_stride %9, %8 : (!cir.ptr<!cir.double>, !s64i) -> !cir.ptr<!cir.double>
    %11 = cir.load align(8) %10 : !cir.ptr<!cir.double>, !cir.double
    %12 = cir.fmul %11, %0 : !cir.double
    cir.store %12, %4 : !cir.double, !cir.ptr<!cir.double>
    %13 = cir.load %4 : !cir.ptr<!cir.double>, !cir.double
    cir.return %13 : !cir.double
  }
}
