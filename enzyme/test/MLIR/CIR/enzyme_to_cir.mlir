// RUN: %eopt --convert-enzyme-to-cir %s | FileCheck %s

// A gradient becomes a stack slot. A cache becomes a stack on the heap: its
// buffer, size and capacity live in stack slots, pushes and pops call helpers
// that grow the buffer with realloc, and the function frees it on return.
module {
  cir.func @f(%x: !cir.double) -> !cir.double {
    %cache = "enzyme.init"() : () -> !enzyme.Cache<!cir.double>
    %grad = "enzyme.init"() : () -> !enzyme.Gradient<!cir.double>
    "enzyme.set"(%grad, %x) : (!enzyme.Gradient<!cir.double>, !cir.double) -> ()
    "enzyme.push"(%cache, %x) : (!enzyme.Cache<!cir.double>, !cir.double) -> ()
    %v = "enzyme.pop"(%cache) : (!enzyme.Cache<!cir.double>) -> !cir.double
    %g = "enzyme.get"(%grad) : (!enzyme.Gradient<!cir.double>) -> !cir.double
    %s = cir.fadd %v, %g : !cir.double
    cir.return %s : !cir.double
  }
}

// CHECK:         cir.func private @free(!cir.ptr<!void>)

// CHECK-LABEL:   cir.func internal private @__enzyme_cir_push(
// CHECK-SAME:      %[[DATA:.+]]: !cir.ptr<!cir.ptr<!cir.double>>, %[[SIZE:.+]]: !cir.ptr<!u64i>, %[[CAP:.+]]: !cir.ptr<!u64i>, %[[V:.+]]: !cir.double) {
// CHECK:           %[[N:.+]] = cir.load %[[SIZE]]
// CHECK:           %[[C:.+]] = cir.load %[[CAP]]
// CHECK:           %[[FULL:.+]] = cir.cmp eq %[[N]], %[[C]] : !u64i
// CHECK:           cir.brcond %[[FULL]] ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[NEWCAP:.+]] = cir.select
// CHECK:           %[[ELEMSIZE:.+]] = cir.const #cir.int<8> : !u64i
// CHECK:           %[[BYTES:.+]] = cir.mul %[[NEWCAP]], %[[ELEMSIZE]] : !u64i
// CHECK:           %[[GROWN:.+]] = cir.call @realloc(%{{.+}}, %[[BYTES]])
// CHECK:           cir.store %{{.+}}, %[[DATA]]
// CHECK:           cir.store %[[NEWCAP]], %[[CAP]]
// CHECK:         ^bb2:
// CHECK:           %[[BUF:.+]] = cir.load %[[DATA]]
// CHECK:           %[[SLOT:.+]] = cir.ptr_stride %[[BUF]], %[[N]]
// CHECK:           cir.store %[[V]], %[[SLOT]] : !cir.double, !cir.ptr<!cir.double>
// CHECK:           %[[N1:.+]] = cir.add %[[N]], %{{.+}} : !u64i
// CHECK:           cir.store %[[N1]], %[[SIZE]]
// CHECK:         cir.func private @realloc(!cir.ptr<!void>, !u64i) -> !cir.ptr<!void>

// CHECK-LABEL:   cir.func internal private @__enzyme_cir_pop(
// CHECK-SAME:      %[[DATA:.+]]: !cir.ptr<!cir.ptr<!cir.double>>, %[[SIZE:.+]]: !cir.ptr<!u64i>, %{{.+}}: !cir.ptr<!u64i>) -> !cir.double {
// CHECK:           %[[N:.+]] = cir.load %[[SIZE]]
// CHECK:           %[[LAST:.+]] = cir.sub %[[N]], %{{.+}} : !u64i
// CHECK:           cir.store %[[LAST]], %[[SIZE]]
// CHECK:           %[[SLOT:.+]] = cir.ptr_stride %{{.+}}, %[[LAST]]
// CHECK:           %[[V:.+]] = cir.load %[[SLOT]] : !cir.ptr<!cir.double>, !cir.double
// CHECK:           cir.return %[[V]]

// CHECK-LABEL:   cir.func @f(
// CHECK-SAME:      %[[X:.+]]: !cir.double) -> !cir.double {
// CHECK-NOT:       enzyme.
// CHECK:           %[[GRAD:.+]] = cir.alloca "enzyme.gradient" align(8) : !cir.ptr<!cir.double>
// CHECK:           %[[DATA:.+]] = cir.alloca "enzyme.cache.data" align(8) : !cir.ptr<!cir.ptr<!cir.double>>
// CHECK:           %[[SIZE:.+]] = cir.alloca "enzyme.cache.size" align(8) : !cir.ptr<!u64i>
// CHECK:           %[[CAP:.+]] = cir.alloca "enzyme.cache.capacity" align(8) : !cir.ptr<!u64i>
// CHECK:           %[[NULL:.+]] = cir.const #cir.ptr<null> : !cir.ptr<!cir.double>
// CHECK:           cir.store %[[NULL]], %[[DATA]]
// CHECK:           cir.store %{{.+}}, %[[SIZE]]
// CHECK:           cir.store %{{.+}}, %[[CAP]]
// CHECK:           cir.store %[[X]], %[[GRAD]] : !cir.double, !cir.ptr<!cir.double>
// CHECK:           cir.call @__enzyme_cir_push(%[[DATA]], %[[SIZE]], %[[CAP]], %[[X]])
// CHECK:           %[[V:.+]] = cir.call @__enzyme_cir_pop(%[[DATA]], %[[SIZE]], %[[CAP]])
// CHECK:           %[[G:.+]] = cir.load %[[GRAD]] : !cir.ptr<!cir.double>, !cir.double
// CHECK:           %[[S:.+]] = cir.fadd %[[V]], %[[G]] : !cir.double
// CHECK:           %[[BUF:.+]] = cir.load %[[DATA]]
// CHECK:           %[[VOID:.+]] = cir.cast bitcast %[[BUF]] : !cir.ptr<!cir.double> -> !cir.ptr<!void>
// CHECK:           cir.call @free(%[[VOID]])
// CHECK-NEXT:      cir.return %[[S]] : !cir.double
