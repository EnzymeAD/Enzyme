// RUN: %eopt --cir-autodiff-pipeline --dump-pass-pipeline %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=PIPELINE
// RUN: %eopt --cir-autodiff-pipeline %s | FileCheck %s --implicit-check-not='{{"enzyme\.[a-z_]+"\(}}' --implicit-check-not=enzyme.autodiff --implicit-check-not=enzyme.fwddiff

// --cir-autodiff-pipeline takes ClangIR as clang emits it at -O0 -- every
// argument spilled to a stack slot, the derivative asked for with a marker --
// to ClangIR with the derivative filled in and no Enzyme ops left.

// PIPELINE:      builtin.module(
// PIPELINE-NEXT:   cir-flatten-cfg,
// PIPELINE-NEXT:   cir-raise-enzyme-calls,
// PIPELINE-NEXT:   mem2reg
// PIPELINE-NEXT:   canonicalize
// PIPELINE-NEXT:   enzyme
// PIPELINE-NEXT:   mem2reg
// PIPELINE-NEXT:   canonicalize
// PIPELINE-NEXT:   convert-enzyme-to-cir

!s32i = !cir.int<s, 32>
!void = !cir.void
module {
  cir.global "private" external @enzyme_const : !s32i

  cir.func @mul(%x: !cir.double, %y: !cir.double) -> !cir.double {
    %xs = cir.alloca "x" align(8) init : !cir.ptr<!cir.double>
    %ys = cir.alloca "y" align(8) init : !cir.ptr<!cir.double>
    cir.store %x, %xs : !cir.double, !cir.ptr<!cir.double>
    cir.store %y, %ys : !cir.double, !cir.ptr<!cir.double>
    %a = cir.load %xs : !cir.ptr<!cir.double>, !cir.double
    %b = cir.load %ys : !cir.ptr<!cir.double>, !cir.double
    %r = cir.fmul %a, %b : !cir.double
    cir.return %r : !cir.double
  }

  cir.func private @__enzyme_autodiff(!cir.ptr<!void>, ...) -> !cir.double

  // d/dx (x * y) with y marked constant.
  cir.func @dmul_dx(%x: !cir.double, %y: !cir.double) -> !cir.double {
    %f = cir.get_global @mul : !cir.ptr<!cir.func<(!cir.double, !cir.double) -> !cir.double>>
    %fp = cir.cast bitcast %f : !cir.ptr<!cir.func<(!cir.double, !cir.double) -> !cir.double>> -> !cir.ptr<!void>
    %cg = cir.get_global @enzyme_const : !cir.ptr<!s32i>
    %c = cir.load %cg : !cir.ptr<!s32i>, !s32i
    %r = cir.call @__enzyme_autodiff(%fp, %x, %c, %y) : (!cir.ptr<!void>, !cir.double, !s32i, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }

  // The gradient is thrown away, as `(void)__enzyme_autodiff(...)` would. The
  // calls are raised before canonicalize, which marks x constant and drops the
  // result, so its derivative has nothing left to compute.
  cir.func @ignored(%x: !cir.double, %y: !cir.double) {
    %f = cir.get_global @mul : !cir.ptr<!cir.func<(!cir.double, !cir.double) -> !cir.double>>
    %fp = cir.cast bitcast %f : !cir.ptr<!cir.func<(!cir.double, !cir.double) -> !cir.double>> -> !cir.ptr<!void>
    %cg = cir.get_global @enzyme_const : !cir.ptr<!s32i>
    %c = cir.load %cg : !cir.ptr<!s32i>, !s32i
    %r = cir.call @__enzyme_autodiff(%fp, %x, %c, %y) : (!cir.ptr<!void>, !cir.double, !s32i, !cir.double) -> !cir.double
    cir.return
  }
}

// The stack slots are gone, the call is the derivative seeded with 1.0, and
// d/dx (x * y) = seed * y with y kept for the backward pass in a lowered cache.
// The derivative whose gradient is thrown away does nothing.
// CHECK-LABEL: cir.func @mul(
// CHECK-NEXT:    cir.fmul
// CHECK-NEXT:    cir.return
// CHECK-LABEL: cir.func @dmul_dx(
// CHECK-SAME:      %[[X:.+]]: !cir.double, %[[Y:.+]]: !cir.double)
// CHECK-NEXT:    %[[ONE:.+]] = cir.const #cir.fp<1.000000e+00> : !cir.double
// CHECK-NEXT:    %[[D:.+]] = cir.call @diffemul(%[[X]], %[[Y]], %[[ONE]])
// CHECK-NEXT:    cir.return %[[D]]
// CHECK-LABEL: cir.func @ignored(
// CHECK-NEXT:    %[[ONE2:.+]] = cir.const #cir.fp<1.000000e+00> : !cir.double
// CHECK-NEXT:    cir.call @[[IGN:diffemul_.+]](%{{.+}}, %{{.+}}, %[[ONE2]]) : (!cir.double, !cir.double, !cir.double) -> ()
// CHECK-NEXT:    cir.return
// CHECK-LABEL: cir.func private @diffemul(
// CHECK-SAME:      %{{.+}}: !cir.double, %[[DY:.+]]: !cir.double, %[[SEED:.+]]: !cir.double)
// CHECK:         cir.call @__enzyme_cir_push(%{{.+}}, %{{.+}}, %{{.+}}, %[[DY]])
// CHECK:         %[[G:.+]] = cir.fadd %{{.+}}, %[[SEED]]
// CHECK:         %[[YP:.+]] = cir.call @__enzyme_cir_pop(
// CHECK:         cir.fmul %[[G]], %[[YP]]
// CHECK:       cir.func private @[[IGN]](
// CHECK-NEXT:    cir.return
// CHECK-NEXT:  }
