// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize %s | FileCheck %s

// A callee with two active arguments returns two gradients packed in a record;
// the caller unpacks them with cir.extract_member. The callee ends in
// cir.return, which must count as touching no memory or reverse mode refuses
// the call.
!rec = !cir.struct<{data !cir.double, data !cir.double}>
module {
  cir.func @g(%x: !cir.double, %y: !cir.double) -> !cir.double {
    %0 = cir.fmul %x, %y : !cir.double
    cir.return %0 : !cir.double
  }
  cir.func @f(%x: !cir.double, %y: !cir.double) -> !cir.double {
    %0 = cir.call @g(%x, %y) : (!cir.double, !cir.double) -> !cir.double
    %1 = cir.sin %0 : !cir.double
    cir.return %1 : !cir.double
  }
  cir.func @df(%x: !cir.double, %y: !cir.double, %dr: !cir.double) -> !rec {
    %res = enzyme.autodiff @f(%x, %y, %dr) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.double, !cir.double) -> !rec
    cir.return %res : !rec
  }
}

// CHECK-LABEL:   cir.func private @diffef(
// CHECK-SAME:      %[[X:[a-zA-Z0-9$._-]+]]: !cir.double,
// CHECK-SAME:      %[[Y:[a-zA-Z0-9$._-]+]]: !cir.double,
// CHECK-SAME:      %[[DR:[a-zA-Z0-9$._-]+]]: !cir.double) -> !rec_anon_struct {
// CHECK:           %[[CALL:.*]] = cir.call @diffeg(%[[X]], %[[Y]], %{{.*}}) : (!cir.double, !cir.double, !cir.double) -> !rec_anon_struct
// CHECK:           %[[DX:.*]] = cir.extract_member %[[CALL]][0] : !rec_anon_struct -> !cir.double
// CHECK:           %[[DY:.*]] = cir.extract_member %[[CALL]][1] : !rec_anon_struct -> !cir.double
// CHECK:           %[[R0:.*]] = cir.insert_member %{{.*}}[0], %{{.*}} : !rec_anon_struct, !cir.double
// CHECK:           cir.insert_member %[[R0]][1], %{{.*}} : !rec_anon_struct, !cir.double

// CHECK-LABEL:   cir.func private @diffeg(
// CHECK-SAME:      -> !rec_anon_struct {
