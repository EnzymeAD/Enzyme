// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s

// @id reads through its pointer, so its derivative is split. @f stores an
// active value through what @id returns, so it needs that pointer's shadow;
// but @id is given %c as constant, and what it returns is inactive there.
// Like an inactive pointer stored into differentiated memory, it has itself
// for its shadow: @augmented_id hands back the primal, not null, which @f
// would write through.

module {
  cir.func @id(%p: !cir.ptr<!cir.double>) -> !cir.ptr<!cir.double> {
    %v = cir.load %p : !cir.ptr<!cir.double>, !cir.double
    cir.return %p : !cir.ptr<!cir.double>
  }
  cir.func @f(%x: !cir.double, %c: !cir.ptr<!cir.double>) -> !cir.double {
    %q = cir.call @id(%c) : (!cir.ptr<!cir.double>) -> !cir.ptr<!cir.double>
    %sq = cir.fmul %x, %x : !cir.double
    cir.store %sq, %q : !cir.double, !cir.ptr<!cir.double>
    %r = cir.load %q : !cir.ptr<!cir.double>, !cir.double
    cir.return %r : !cir.double
  }
  cir.func @rev(%x: !cir.double, %c: !cir.ptr<!cir.double>, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @f(%x, %c, %seed) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.ptr<!cir.double>, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// CHECK-LABEL: cir.func private @diffef(
// CHECK:         %[[R:.+]] = cir.call @augmented_id(%{{.+}}) : (!cir.ptr<!cir.double>) -> !rec_anon_struct
// CHECK:         %[[DQ:.+]] = cir.extract_member %[[R]][1]
// CHECK-LABEL: cir.func private @augmented_id(
// CHECK-SAME:    %[[P:[^:]+]]: !cir.ptr<!cir.double>) -> !rec_anon_struct
// CHECK:         cir.insert_member %{{.+}}[0], %[[P]]
// CHECK-NEXT:    cir.insert_member %{{.+}}[1], %[[P]]
