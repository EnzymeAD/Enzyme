// RUN: %eopt --convert-enzyme-to-cir %s | FileCheck %s

// A gradient becomes a stack slot. A cache becomes a stack: where its buffer
// is, its size and its capacity live in stack slots, and the buffer starts as
// an inline one of 8 values in the frame. Pushes and pops call helpers; a
// push that finds the buffer full doubles it, copying the inline buffer to
// the heap the first time and growing it with realloc after, and the
// function frees a heap buffer on return.
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

// CHECK-LABEL:   cir.func internal private @__enzyme_cir_release(
// CHECK-SAME:      %[[DATA:.+]]: !cir.ptr<!cir.ptr<!cir.double>>, %[[INLINE:.+]]: !cir.ptr<!cir.double>) {
// CHECK:           %[[BUF:.+]] = cir.load %[[DATA]]
// CHECK:           %[[HEAP:.+]] = cir.cmp ne %[[BUF]], %[[INLINE]] : !cir.ptr<!cir.double>
// CHECK:           cir.brcond %[[HEAP]] ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           cir.call @free(
// CHECK:         ^bb2:
// CHECK:           cir.return

// CHECK-LABEL:   cir.func internal private @__enzyme_cir_push(
// CHECK-SAME:      %[[DATA:.+]]: !cir.ptr<!cir.ptr<!cir.double>>, %[[SIZE:.+]]: !cir.ptr<!u64i>, %[[CAP:.+]]: !cir.ptr<!u64i>, %[[INLINE:.+]]: !cir.ptr<!cir.double>, %[[V:.+]]: !cir.double) {
// CHECK:           %[[N:.+]] = cir.load %[[SIZE]]
// CHECK:           %[[C:.+]] = cir.load %[[CAP]]
// CHECK:           %[[FULL:.+]] = cir.cmp eq %[[N]], %[[C]] : !u64i
// CHECK:           cir.brcond %[[FULL]] ^bb1, ^bb4
// CHECK:         ^bb1:
// CHECK:           %[[NEWCAP:.+]] = cir.select
// CHECK:           %[[ELEMSIZE:.+]] = cir.const #cir.int<8> : !u64i
// CHECK:           %[[BYTES:.+]] = cir.mul %[[NEWCAP]], %[[ELEMSIZE]] : !u64i
// CHECK:           cir.store %[[NEWCAP]], %[[CAP]]
// CHECK:           %[[OLD:.+]] = cir.load %[[DATA]]
// CHECK:           %[[ININLINE:.+]] = cir.cmp eq %[[OLD]], %[[INLINE]]
// CHECK:           cir.brcond %[[ININLINE]] ^bb2, ^bb3
// CHECK:         ^bb2:
// CHECK:           %[[NEW:.+]] = cir.call @malloc(%[[BYTES]])
// CHECK:           cir.libc.memcpy %{{.+}} bytes from %{{.+}} to %[[NEW]]
// CHECK:           cir.store %{{.+}}, %[[DATA]]
// CHECK:         ^bb3:
// CHECK:           cir.call @realloc(%{{.+}}, %[[BYTES]])
// CHECK:           cir.store %{{.+}}, %[[DATA]]
// CHECK:         ^bb4:
// CHECK:           %[[BUF:.+]] = cir.load %[[DATA]]
// CHECK:           %[[SLOT:.+]] = cir.ptr_stride %[[BUF]], %[[N]]
// CHECK:           cir.store %[[V]], %[[SLOT]] : !cir.double, !cir.ptr<!cir.double>
// CHECK:           %[[N1:.+]] = cir.add %[[N]], %{{.+}} : !u64i
// CHECK:           cir.store %[[N1]], %[[SIZE]]
// CHECK:         cir.func private @realloc(!cir.ptr<!void>, !u64i) -> !cir.ptr<!void>
// CHECK:         cir.func private @malloc(!u64i) -> !cir.ptr<!void>

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
// CHECK:           %[[INLINEBUF:.+]] = cir.alloca "enzyme.cache.inline" align(8) : !cir.ptr<!cir.array<!cir.double x 8>>
// CHECK:           %[[INLINE:.+]] = cir.cast array_to_ptrdecay %[[INLINEBUF]]
// CHECK:           cir.store %[[INLINE]], %[[DATA]]
// CHECK:           cir.store %{{.+}}, %[[SIZE]]
// CHECK:           %[[EIGHT:.+]] = cir.const #cir.int<8> : !u64i
// CHECK:           cir.store %[[EIGHT]], %[[CAP]]
// CHECK:           cir.store %[[X]], %[[GRAD]] : !cir.double, !cir.ptr<!cir.double>
// CHECK:           cir.call @__enzyme_cir_push(%[[DATA]], %[[SIZE]], %[[CAP]], %[[INLINE]], %[[X]])
// CHECK:           %[[V:.+]] = cir.call @__enzyme_cir_pop(%[[DATA]], %[[SIZE]], %[[CAP]])
// CHECK:           %[[G:.+]] = cir.load %[[GRAD]] : !cir.ptr<!cir.double>, !cir.double
// CHECK:           %[[S:.+]] = cir.fadd %[[V]], %[[G]] : !cir.double
// CHECK:           cir.call @__enzyme_cir_release(%[[DATA]], %[[INLINE]])
// CHECK-NEXT:      cir.return %[[S]] : !cir.double
