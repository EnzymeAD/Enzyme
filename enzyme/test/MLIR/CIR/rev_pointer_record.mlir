// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s

// A record of pointers is a pointer to its shadow much as each of its
// pointers is: its shadow is the record of their shadows, which the forward
// sweep carries along when the record is copied, rather than the reverse
// sweep adding it up. The adjoints of what its pointers point to are then
// found through the copy.

!rec = !cir.struct<{data !cir.ptr<!cir.double>, data !cir.ptr<!cir.double>}>
module {
  cir.func @pair(%a: !cir.ptr<!cir.double>, %b: !cir.ptr<!cir.double>) -> !cir.double {
    %s = cir.alloca "s" align(8) : !cir.ptr<!rec>
    %c = cir.alloca "c" align(8) : !cir.ptr<!rec>
    %sa = cir.get_member %s[0] {name = "a"} : !cir.ptr<!rec> -> !cir.ptr<!cir.ptr<!cir.double>>
    cir.store %a, %sa : !cir.ptr<!cir.double>, !cir.ptr<!cir.ptr<!cir.double>>
    %sb = cir.get_member %s[1] {name = "b"} : !cir.ptr<!rec> -> !cir.ptr<!cir.ptr<!cir.double>>
    cir.store %b, %sb : !cir.ptr<!cir.double>, !cir.ptr<!cir.ptr<!cir.double>>
    %v = cir.load %s : !cir.ptr<!rec>, !rec
    cir.store %v, %c : !rec, !cir.ptr<!rec>
    %ca = cir.get_member %c[0] {name = "a"} : !cir.ptr<!rec> -> !cir.ptr<!cir.ptr<!cir.double>>
    %pa = cir.load %ca : !cir.ptr<!cir.ptr<!cir.double>>, !cir.ptr<!cir.double>
    %cb = cir.get_member %c[1] {name = "b"} : !cir.ptr<!rec> -> !cir.ptr<!cir.ptr<!cir.double>>
    %pb = cir.load %cb : !cir.ptr<!cir.ptr<!cir.double>>, !cir.ptr<!cir.double>
    %x = cir.load %pa : !cir.ptr<!cir.double>, !cir.double
    %y = cir.load %pb : !cir.ptr<!cir.double>, !cir.double
    %r = cir.fmul %x, %y : !cir.double
    cir.return %r : !cir.double
  }
  cir.func @rev(%a: !cir.ptr<!cir.double>, %da: !cir.ptr<!cir.double>, %b: !cir.ptr<!cir.double>, %db: !cir.ptr<!cir.double>, %seed: !cir.double) {
    enzyme.autodiff @pair(%a, %da, %b, %db, %seed) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !cir.double) -> ()
    cir.return
  }
}

// The shadow record goes into the copy's shadow in the forward sweep, and the
// shadow pointers are read back from there.
// CHECK-LABEL: cir.func private @diffepair(
// CHECK-SAME:    %[[A:[^:]+]]: !cir.ptr<!cir.double>, %[[DA:[^:]+]]: !cir.ptr<!cir.double>, %[[B:[^:]+]]: !cir.ptr<!cir.double>, %[[DB:[^:]+]]: !cir.ptr<!cir.double>
// CHECK:         %[[DS:.+]] = cir.alloca "s"
// CHECK:         %[[DC:.+]] = cir.alloca "c"
// CHECK:         %[[DSA:.+]] = cir.get_member %[[DS]][0]
// CHECK:         cir.store %[[DA]], %[[DSA]]
// CHECK:         %[[DSB:.+]] = cir.get_member %[[DS]][1]
// CHECK:         cir.store %[[DB]], %[[DSB]]
// CHECK:         %[[DV:.+]] = cir.load %[[DS]] : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CHECK:         cir.store %[[DV]], %[[DC]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
// CHECK:         %[[DCA:.+]] = cir.get_member %[[DC]][0]
// CHECK:         %[[DPA:.+]] = cir.load %[[DCA]] : !cir.ptr<!cir.ptr<!cir.double>>, !cir.ptr<!cir.double>
// CHECK:         %[[DCB:.+]] = cir.get_member %[[DC]][1]
// CHECK:         %[[DPB:.+]] = cir.load %[[DCB]] : !cir.ptr<!cir.ptr<!cir.double>>, !cir.ptr<!cir.double>
// CHECK:         "enzyme.push"(%[[PACACHE:.+]], %[[DPA]])
// CHECK:         "enzyme.push"(%[[PBCACHE:.+]], %[[DPB]])

// The reverse sweep accumulates through them, and adds nothing up for the
// record itself.
// CHECK:         %[[RB:.+]] = "enzyme.pop"(%[[PBCACHE]])
// CHECK-NEXT:    %[[LB:.+]] = cir.load %[[RB]]
// CHECK-NEXT:    %[[SB:.+]] = cir.fadd %[[LB]]
// CHECK-NEXT:    cir.store %[[SB]], %[[RB]]
// CHECK:         %[[RA:.+]] = "enzyme.pop"(%[[PACACHE]])
// CHECK-NEXT:    %[[LA:.+]] = cir.load %[[RA]]
// CHECK-NEXT:    %[[SA:.+]] = cir.fadd %[[LA]]
// CHECK-NEXT:    cir.store %[[SA]], %[[RA]]
// CHECK-NOT:     cir.extract_member
// CHECK:         cir.return
