// RUN: %eopt --enzyme %s | FileCheck %s

// The derivative of a call returns the primal and the tangent, but a cir.func
// returns one value: they come back packed in a record and are unpacked with
// cir.extract_member at the call site.
module {
  cir.func @g(%x: !cir.double, %y: !cir.double) -> !cir.double {
    %0 = cir.fmul %x, %y : !cir.double
    cir.return %0 : !cir.double
  }
  cir.func @f(%x: !cir.double, %y: !cir.double) -> !cir.double {
    %0 = cir.call @g(%x, %y) : (!cir.double, !cir.double) -> !cir.double
    %1 = cir.fmul %0, %x : !cir.double
    cir.return %1 : !cir.double
  }
  cir.func @df(%x: !cir.double, %dx: !cir.double, %y: !cir.double) -> !cir.double {
    %r = enzyme.fwddiff @f(%x, %dx, %y) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (!cir.double, !cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// CHECK-LABEL:   cir.func private @fwddiffef(
// CHECK-SAME:      %[[X:[a-zA-Z0-9$._-]+]]: !cir.double,
// CHECK-SAME:      %[[DX:[a-zA-Z0-9$._-]+]]: !cir.double,
// CHECK-SAME:      %[[Y:[a-zA-Z0-9$._-]+]]: !cir.double) -> !cir.double {
// CHECK:           %[[CALL:.*]] = cir.call @fwddiffeg(%[[X]], %[[DX]], %[[Y]]) : (!cir.double, !cir.double, !cir.double) -> !rec_anon_struct
// CHECK:           %[[G:.*]] = cir.extract_member %[[CALL]][0] : !rec_anon_struct -> !cir.double
// CHECK:           %[[DG:.*]] = cir.extract_member %[[CALL]][1] : !rec_anon_struct -> !cir.double
// CHECK:           %[[T0:.*]] = cir.fmul %[[DG]], %[[X]] : !cir.double
// CHECK:           %[[T1:.*]] = cir.fmul %[[DX]], %[[G]] : !cir.double
// CHECK:           %[[DF:.*]] = cir.fadd %[[T0]], %[[T1]] : !cir.double
// CHECK:           cir.return %[[DF]] : !cir.double

// CHECK-LABEL:   cir.func private @fwddiffeg(
// CHECK-SAME:      -> !rec_anon_struct {
// CHECK:           %[[PACK0:.*]] = cir.insert_member %{{.*}}[0], %{{.*}} : !rec_anon_struct, !cir.double
// CHECK:           %[[PACK1:.*]] = cir.insert_member %[[PACK0]][1], %{{.*}} : !rec_anon_struct, !cir.double
// CHECK:           cir.return %[[PACK1]] : !rec_anon_struct
