// RUN: %eopt --enzyme %s | FileCheck %s
// RUN: %eopt --enzyme --convert-enzyme-to-cir %s | FileCheck %s --check-prefix=LOWER

// A callee that touches memory cannot run its primal again in the reverse
// sweep: @scale's store would be repeated on what it already scaled. Its
// derivative is split instead. @augmented_scale runs the primal and the
// forward sweep and pushes what the reverse sweep needs onto a tape the
// caller owns; @diffescale only runs the reverse sweep, off that tape. A
// callee returning a pointer returns its shadow too.

!s32i = !cir.int<s, 32>
module {
  cir.func @scale(%p: !cir.ptr<!cir.double>, %a: !cir.double) {
    %v = cir.load %p : !cir.ptr<!cir.double>, !cir.double
    %w = cir.fmul %v, %a : !cir.double
    cir.store %w, %p : !cir.double, !cir.ptr<!cir.double>
    cir.return
  }
  cir.func @pick(%p: !cir.ptr<!cir.double>, %k: !s32i) -> !cir.ptr<!cir.double> {
    %r = cir.ptr_stride %p, %k : (!cir.ptr<!cir.double>, !s32i) -> !cir.ptr<!cir.double>
    %v = cir.load %r : !cir.ptr<!cir.double>, !cir.double
    %w = cir.fmul %v, %v : !cir.double
    cir.store %w, %r : !cir.double, !cir.ptr<!cir.double>
    cir.return %r : !cir.ptr<!cir.double>
  }
  cir.func @f(%x: !cir.double, %k: !s32i) -> !cir.double {
    %zero = cir.const #cir.int<0> : !s32i
    %one = cir.const #cir.int<1> : !s32i
    %v = cir.alloca "v" align(8) : !cir.ptr<!cir.array<!cir.double x 2>>
    %s = cir.cast array_to_ptrdecay %v : !cir.ptr<!cir.array<!cir.double x 2>> -> !cir.ptr<!cir.double>
    %t = cir.ptr_stride %s, %one : (!cir.ptr<!cir.double>, !s32i) -> !cir.ptr<!cir.double>
    cir.store %x, %s : !cir.double, !cir.ptr<!cir.double>
    cir.store %x, %t : !cir.double, !cir.ptr<!cir.double>
    cir.call @scale(%s, %x) : (!cir.ptr<!cir.double>, !cir.double) -> ()
    %p = cir.call @pick(%s, %k) : (!cir.ptr<!cir.double>, !s32i) -> !cir.ptr<!cir.double>
    cir.call @scale(%p, %x) : (!cir.ptr<!cir.double>, !cir.double) -> ()
    %a = cir.load %s : !cir.ptr<!cir.double>, !cir.double
    %b = cir.load %t : !cir.ptr<!cir.double>, !cir.double
    %r = cir.fadd %a, %b : !cir.double
    cir.return %r : !cir.double
  }
  cir.func @df(%x: !cir.double, %k: !s32i) -> !cir.double {
    %one = cir.const #cir.fp<1.0> : !cir.double
    %d = enzyme.autodiff @f(%x, %k, %one) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !s32i, !cir.double) -> !cir.double
    cir.return %d : !cir.double
  }
}

// Each call has its own tape, which the caller passes as pointers to the
// augmented call in the forward sweep and to the reverse call in the reverse
// sweep; the reverse calls come in the opposite order. @augmented_pick
// returns the pointer and its shadow.
// CHECK-LABEL: cir.func private @diffef(
// CHECK:         cir.call @augmented_scale(%{{[^,]+}}, %{{[^,]+}}, %arg0, %[[S1A:[^,]+]], %[[S1B:[^,]+]], %[[S1C:[^,]+]], %[[S1D:[^)]+]])
// CHECK:         %[[PICK:.+]] = cir.call @augmented_pick(%{{[^,]+}}, %{{[^,]+}}, %arg1, %[[PA:[^,]+]], %[[PB:[^,]+]], %[[PC:[^,]+]], %[[PD:[^)]+]])
// CHECK-NEXT:    %[[P:.+]] = cir.extract_member %[[PICK]][0]
// CHECK-NEXT:    %[[DP:.+]] = cir.extract_member %[[PICK]][1]
// CHECK:         cir.call @augmented_scale(%[[P]], %[[DP]], %arg0, %[[S2A:[^,]+]], %[[S2B:[^,]+]], %[[S2C:[^,]+]], %[[S2D:[^)]+]])
// CHECK:         cir.call @diffescale(%{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %[[S2A]], %[[S2B]], %[[S2C]], %[[S2D]])
// CHECK:         cir.call @diffepick(%{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %[[PA]], %[[PB]], %[[PC]], %[[PD]])
// CHECK:         cir.call @diffescale(%{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %[[S1A]], %[[S1B]], %[[S1C]], %[[S1D]])

// The augmented function runs the primal and pushes.
// CHECK-LABEL: cir.func private @augmented_scale(
// CHECK-SAME:      %[[P:[^:]+]]: !cir.ptr<!cir.double>, %{{[^:]+}}: !cir.ptr<!cir.double>, %{{[^:]+}}: !cir.double, %{{[^:]+}}: !cir.ptr<!void>, %{{[^:]+}}: !cir.ptr<!void>, %{{[^:]+}}: !cir.ptr<!void>, %{{[^:]+}}: !cir.ptr<!void>) {
// CHECK:         "enzyme.push"
// CHECK:         cir.store %{{.+}}, %[[P]]
// CHECK-NEXT:    cir.return

// The reverse function only pops: the primal store is not run again.
// CHECK-LABEL: cir.func private @diffescale(
// CHECK-SAME:      %[[P:[^:]+]]: !cir.ptr<!cir.double>,
// CHECK-NOT:     cir.store %{{.+}}, %[[P]]
// CHECK:         "enzyme.pop"
// CHECK-NOT:     cir.store %{{.+}}, %[[P]]
// CHECK:         cir.return

// CHECK-LABEL: cir.func private @augmented_pick(
// CHECK:         %[[DR:.+]] = cir.ptr_stride %arg1, %arg2
// CHECK-NEXT:    %[[R:.+]] = cir.ptr_stride %arg0, %arg2
// CHECK:         %[[A:.+]] = cir.insert_member %{{.+}}[0], %[[R]]
// CHECK-NEXT:    %[[B:.+]] = cir.insert_member %[[A]][1], %[[DR]]
// CHECK-NEXT:    cir.return %[[B]]

// Lowered, a tape cache the caller passes is a record of its three fields,
// which the caller frees and the callee uses through the pointer.
// LOWER-NOT:     unrealized_conversion_cast
// LOWER-LABEL: cir.func private @diffef(
// LOWER:         %[[REC:.+]] = cir.alloca "enzyme.cache" align(8) : !cir.ptr<![[RT:.+]]>
// LOWER:         cir.get_member %[[REC]][0] {name = "enzyme.cache.data"}
// LOWER:         cir.call @free(
// LOWER-LABEL: cir.func private @diffescale(
// LOWER-SAME:      %arg3: !cir.ptr<!void>
// LOWER:         %[[T:.+]] = cir.cast bitcast %arg3 : !cir.ptr<!void> -> !cir.ptr<!
// LOWER-NEXT:    cir.get_member %[[T]][0] {name = "enzyme.cache.data"}
// LOWER-NOT:     cir.call @free(
// LOWER:         cir.return
