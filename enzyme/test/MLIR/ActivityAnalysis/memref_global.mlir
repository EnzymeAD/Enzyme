// RUN: %eopt --print-activity-analysis='dataflow=false annotate=true' %s 2>&1 | FileCheck %s

// A global without enzyme_shadow is inactive if it is private and nothing in
// the module either stores anything active into it or actively loads from
// it. A public global could be accessed outside this module, so it is active.

module {
  memref.global "private" @priv_wo : memref<f64> = dense<0.0>
  memref.global @pub_wo : memref<f64> = dense<0.0>
  memref.global "private" @priv_other : memref<f64> = dense<0.0>
  memref.global "private" @priv_rw : memref<f64> = dense<0.0>
  memref.global "private" @priv_callee : memref<f64> = dense<0.0>
  memref.global "private" @priv_const : memref<f64> = dense<0.0>
  memref.global @ro : memref<f64> = dense<2.0>
  memref.global @marked : memref<f64> = dense<0.0> {enzyme_inactive}
  memref.global @count : memref<i32> = dense<0>
  memref.global "private" @logged : memref<f64> = dense<0.0>
  memref.global "private" @passed : memref<f64> = dense<0.0>
  memref.global "private" @arg_callee : memref<f64> = dense<0.0>
  memref.global "private" @arg_call : memref<f64> = dense<0.0>
  memref.global "private" @arg_other : memref<f64> = dense<0.0>
  memref.global "private" @priv_loop : memref<f64> = dense<0.0>
  memref.global "private" @nc_read : memref<f64> = dense<0.0>
  memref.global "private" @nc_write : memref<f64> = dense<0.0>
  memref.global "private" @nc_effects : memref<f64> = dense<0.0>
  memref.global "private" @ro_capture : memref<f64> = dense<0.0>
  memref.global @bytes : memref<8xi8> = dense<0>

  func.func @write_only_private(%x: f64) -> f64 {
    %p = memref.get_global @priv_wo : memref<f64>
    memref.store %x, %p[] : memref<f64>
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  func.func @write_only_public(%x: f64) -> f64 {
    %p = memref.get_global @pub_wo : memref<f64>
    memref.store %x, %p[] : memref<f64>
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  func.func @read_other() -> f64 {
    %p = memref.get_global @priv_other : memref<f64>
    %v = memref.load %p[] : memref<f64>
    return %v : f64
  }
  func.func @write_read_elsewhere(%x: f64) -> f64 {
    %p = memref.get_global @priv_other : memref<f64>
    memref.store %x, %p[] : memref<f64>
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  func.func @write_read_here(%x: f64) -> f64 {
    %p = memref.get_global @priv_rw : memref<f64>
    memref.store %x, %p[] : memref<f64>
    %q = memref.get_global @priv_rw : memref<f64>
    %v = memref.load %q[] : memref<f64>
    %r = arith.mulf %v, %v : f64
    return %r : f64
  }
  func.func private @set_callee(%x: f64) {
    %p = memref.get_global @priv_callee : memref<f64>
    memref.store %x, %p[] : memref<f64>
    return
  }
  func.func @read_after_callee(%x: f64) -> f64 {
    call @set_callee(%x) : (f64) -> ()
    %p = memref.get_global @priv_callee : memref<f64>
    %v = memref.load %p[] : memref<f64>
    return %v : f64
  }
  func.func private @set_const() {
    %c = arith.constant 3.0 : f64
    %p = memref.get_global @priv_const : memref<f64>
    memref.store %c, %p[] : memref<f64>
    return
  }
  func.func @read_const(%x: f64) -> f64 {
    %p = memref.get_global @priv_const : memref<f64>
    %v = memref.load %p[] : memref<f64>
    %r = arith.mulf %x, %v : f64
    return %r : f64
  }
  func.func @marked_inactive(%x: f64) -> f64 {
    %p = memref.get_global @marked : memref<f64>
    memref.store %x, %p[] : memref<f64>
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  func.func @read_only_public(%x: f64) -> f64 {
    %p = memref.get_global @ro : memref<f64>
    %v = memref.load %p[] : memref<f64>
    %r = arith.mulf %x, %v : f64
    return %r : f64
  }
  func.func @loop_carried(%x: f64) -> f64 {
    %p = memref.get_global @priv_loop : memref<f64>
    memref.store %x, %p[] : memref<f64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %init = arith.constant 0.0 : f64
    %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %init) -> (f64) {
      %t = arith.addf %acc, %acc : f64
      %q = memref.get_global @priv_loop : memref<f64>
      %v = memref.load %q[] : memref<f64>
      %s = arith.addf %t, %v : f64
      scf.yield %s : f64
    }
    return %r : f64
  }
  func.func private @print_buf(memref<f64>) attributes {enzyme_inactive}
  func.func private @ext(memref<f64>)
  func.func @inactive_call(%x: f64) -> f64 {
    %p = memref.get_global @logged : memref<f64>
    memref.store %x, %p[] : memref<f64>
    call @print_buf(%p) : (memref<f64>) -> ()
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  func.func @unknown_call(%x: f64) -> f64 {
    %p = memref.get_global @passed : memref<f64>
    memref.store %x, %p[] : memref<f64>
    call @ext(%p) : (memref<f64>) -> ()
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  func.func private @scale(%v: f64, %m: memref<f64> {enzyme.inactive}) -> f64
  func.func private @scale_plain(%v: f64, %m: memref<f64>) -> f64
  func.func private @scale_other(%v: f64 {enzyme.inactive}, %m: memref<f64>) -> f64
  func.func @arg_marked_on_callee(%x: f64) -> f64 {
    %p = memref.get_global @arg_callee : memref<f64>
    memref.store %x, %p[] : memref<f64>
    %r = call @scale(%x, %p) : (f64, memref<f64>) -> f64
    return %r : f64
  }
  func.func @arg_marked_on_call(%x: f64) -> f64 {
    %p = memref.get_global @arg_call : memref<f64>
    memref.store %x, %p[] : memref<f64>
    %r = "func.call"(%x, %p) <{callee = @scale_plain, arg_attrs = [{}, {enzyme.inactive}]}> : (f64, memref<f64>) -> f64
    return %r : f64
  }
  func.func @other_arg_marked(%x: f64) -> f64 {
    %p = memref.get_global @arg_other : memref<f64>
    memref.store %x, %p[] : memref<f64>
    %r = call @scale_other(%x, %p) : (f64, memref<f64>) -> f64
    return %r : f64
  }
  func.func private @peek(%m: memref<f64> {llvm.nocapture, llvm.readonly}) -> f64
  func.func private @clear(%m: memref<f64> {llvm.nocapture, llvm.writeonly})
  func.func private @peek_effects(%m: memref<f64> {llvm.nocapture}) -> f64 attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>}
  func.func private @peek_capture(%m: memref<f64> {llvm.readonly}) -> f64
  func.func @nocapture_read(%x: f64) -> f64 {
    %c = arith.constant 3.0 : f64
    %p = memref.get_global @nc_read : memref<f64>
    memref.store %c, %p[] : memref<f64>
    %v = call @peek(%p) : (memref<f64>) -> f64
    %r = arith.mulf %x, %v : f64
    return %r : f64
  }
  func.func @nocapture_write(%x: f64) -> f64 {
    %p = memref.get_global @nc_write : memref<f64>
    memref.store %x, %p[] : memref<f64>
    call @clear(%p) : (memref<f64>) -> ()
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  func.func @nocapture_effects(%x: f64) -> f64 {
    %c = arith.constant 3.0 : f64
    %p = memref.get_global @nc_effects : memref<f64>
    memref.store %c, %p[] : memref<f64>
    %v = call @peek_effects(%p) : (memref<f64>) -> f64
    %r = arith.mulf %x, %v : f64
    return %r : f64
  }
  func.func @readonly_capture(%x: f64) -> f64 {
    %c = arith.constant 3.0 : f64
    %p = memref.get_global @ro_capture : memref<f64>
    memref.store %c, %p[] : memref<f64>
    %v = call @peek_capture(%p) : (memref<f64>) -> f64
    %r = arith.mulf %x, %v : f64
    return %r : f64
  }
  func.func private @use_count(memref<i32>)
  func.func @int_global(%x: f64) -> f64 {
    %p = memref.get_global @count : memref<i32>
    call @use_count(%p) : (memref<i32>) -> ()
    return %x : f64
  }
  func.func private @use_bytes(memref<8xi8>)
  func.func @byte_global(%x: f64) -> f64 {
    %p = memref.get_global @bytes : memref<8xi8>
    call @use_bytes(%p) : (memref<8xi8>) -> ()
    return %x : f64
  }
}

// A private global that is only written cannot hand the value back.
// CHECK-LABEL: func.func @write_only_private
// CHECK:         memref.get_global @priv_wo {{.*}}enzyme.res_icv0 = true

// A public one could be read outside this module.
// CHECK-LABEL: func.func @write_only_public
// CHECK:         memref.get_global @pub_wo {{.*}}enzyme.res_icv0 = false

// @write_read_elsewhere stores an active value, and this load returns it.
// CHECK-LABEL: func.func @read_other
// CHECK:         memref.get_global @priv_other {{.*}}enzyme.res_icv0 = false

// @read_other loads the value back.
// CHECK-LABEL: func.func @write_read_elsewhere
// CHECK:         memref.get_global @priv_other {{.*}}enzyme.res_icv0 = false

// The store and the load go through different addresses of the same global;
// both are active.
// CHECK-LABEL: func.func @write_read_here
// CHECK:         memref.get_global @priv_rw {{.*}}enzyme.res_icv0 = false
// CHECK:         memref.get_global @priv_rw {{.*}}enzyme.res_icv0 = false

// The callee stores an active value that is loaded back here.
// CHECK-LABEL: func.func @read_after_callee
// CHECK:         memref.get_global @priv_callee {{.*}}enzyme.res_icv0 = false

// Only a constant is ever stored.
// CHECK-LABEL: func.func @read_const
// CHECK:         memref.get_global @priv_const {{.*}}enzyme.res_icv0 = true

// The user asserts that the global is inactive.
// CHECK-LABEL: func.func @marked_inactive
// CHECK:         memref.get_global @marked {{.*}}enzyme.res_icv0 = true

// Another module could store an active value into it.
// CHECK-LABEL: func.func @read_only_public
// CHECK:         memref.get_global @ro {{.*}}enzyme.res_icv0 = false

// It holds only integers.
// CHECK-LABEL: func.func @int_global
// CHECK:         memref.get_global @count {{.*}}enzyme.res_icv0 = true

// A byte buffer may hold anything, e.g. through memref.view.
// CHECK-LABEL: func.func @byte_global
// CHECK:         memref.get_global @bytes {{.*}}enzyme.res_icv0 = false

// %t is analyzed first, and its UP proof reaches %q through the loop-carried
// %acc before the top level does. Inside that hypothesis %q must not be proven
// inactive on the hypothesis's own assumption that the load is inactive.
// CHECK-LABEL: func.func @loop_carried
// CHECK:         memref.get_global @priv_loop {{.*}}enzyme.res_icv0 = false
// CHECK:         memref.get_global @priv_loop {{.*}}enzyme.res_icv0 = false
// CHECK-NEXT:    memref.load {{.*}}enzyme.res_icv0 = false

// A call known to be inactive neither reads nor writes active data, so the
// only access that matters is the store, and nothing loads it back.
// CHECK-LABEL: func.func @inactive_call
// CHECK:         memref.get_global @logged {{.*}}enzyme.res_icv0 = true

// An unknown call may read the active value back out.
// CHECK-LABEL: func.func @unknown_call
// CHECK:         memref.get_global @passed {{.*}}enzyme.res_icv0 = false

// The global is passed only as an argument marked enzyme.inactive, on the
// callee or on the call, so the call does not move its data, even though the
// call's result is active through %x.
// CHECK-LABEL: func.func @arg_marked_on_callee
// CHECK:         memref.get_global @arg_callee {{.*}}enzyme.res_icv0 = true
// CHECK-LABEL: func.func @arg_marked_on_call
// CHECK:         memref.get_global @arg_call {{.*}}enzyme.res_icv0 = true

// Another argument is marked, not the global's.
// CHECK-LABEL: func.func @other_arg_marked
// CHECK:         memref.get_global @arg_other {{.*}}enzyme.res_icv0 = false

// A call that keeps no copy of the address can only store into the global
// through it if it may write, and only read it out if it may read. Only a
// constant is ever stored into @nc_read and @nc_effects, and @peek and
// @peek_effects only read; @clear only writes @nc_write, and nothing reads it.
// CHECK-LABEL: func.func @nocapture_read
// CHECK:         memref.get_global @nc_read {{.*}}enzyme.res_icv0 = true
// CHECK-LABEL: func.func @nocapture_write
// CHECK:         memref.get_global @nc_write {{.*}}enzyme.res_icv0 = true
// CHECK-LABEL: func.func @nocapture_effects
// CHECK:         memref.get_global @nc_effects {{.*}}enzyme.res_icv0 = true

// @peek_capture only reads, but may keep the address and write through it
// later.
// CHECK-LABEL: func.func @readonly_capture
// CHECK:         memref.get_global @ro_capture {{.*}}enzyme.res_icv0 = false
