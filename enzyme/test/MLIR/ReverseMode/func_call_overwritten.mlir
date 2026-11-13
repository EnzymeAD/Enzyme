// RUN: %eopt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

module {
  // 1. Read-only callee, caller does NOT overwrite %a -> no clone needed.
  func.func @ro(%a: memref<f32>, %b: f32) -> f32 {
    %v = memref.load %a[] : memref<f32>
    %r = arith.mulf %v, %b : f32
    return %r : f32
  }
  func.func @caller_ro_clean(%a: memref<f32>, %b: f32) -> f32 {
    %r = func.call @ro(%a, %b) : (memref<f32>, f32) -> f32
    return %r : f32
  }
  func.func @d_caller_ro_clean(%a: memref<f32>, %da: memref<f32>, %b: f32, %dr: f32) -> f32 {
    %0 = enzyme.autodiff @caller_ro_clean(%a, %da, %b, %dr) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (memref<f32>, memref<f32>, f32, f32) -> f32
    return %0 : f32
  }

  // 2. Read-only callee, caller overwrites %a after call -> clone required.
  func.func @caller_ro_overwritten(%a: memref<f32>, %b: f32) -> f32 {
    %r = func.call @ro(%a, %b) : (memref<f32>, f32) -> f32
    memref.store %b, %a[] : memref<f32>
    return %r : f32
  }
  func.func @d_caller_ro_overwritten(%a: memref<f32>, %da: memref<f32>, %b: f32, %dr: f32) -> f32 {
    %0 = enzyme.autodiff @caller_ro_overwritten(%a, %da, %b, %dr) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (memref<f32>, memref<f32>, f32, f32) -> f32
    return %0 : f32
  }

  // 3. Same SSA operand passed twice when callee writes one slot -> clone once
  //    and pass the same clone to both slots in @diffealias_same, then free once.
  func.func @alias_same(%a: memref<f32>, %b: memref<f32>, %x: f32) -> f32 {
    memref.store %x, %a[] : memref<f32>
    %v = memref.load %b[] : memref<f32>
    return %v : f32
  }
  func.func @caller_alias_same(%a: memref<f32>, %x: f32) -> f32 {
    %r = func.call @alias_same(%a, %a, %x) : (memref<f32>, memref<f32>, f32) -> f32
    return %r : f32
  }
  func.func @d_caller_alias_same(%a: memref<f32>, %da: memref<f32>, %x: f32, %dr: f32) -> f32 {
    %0 = enzyme.autodiff @caller_alias_same(%a, %da, %x, %dr) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (memref<f32>, memref<f32>, f32, f32) -> f32
    return %0 : f32
  }

  // 4. Constant (non-dup) memref written by callee -> still cloned and freed.
  func.func @write_const(%a: memref<f32>, %x: f32) -> f32 {
    %old = memref.load %a[] : memref<f32>
    %sum = arith.addf %old, %x : f32
    memref.store %sum, %a[] : memref<f32>
    %v = memref.load %a[] : memref<f32>
    %sq = arith.mulf %v, %x : f32
    return %sq : f32
  }
  func.func @caller_write_const(%a: memref<f32>, %x: f32) -> f32 {
    %r = func.call @write_const(%a, %x) : (memref<f32>, f32) -> f32
    return %r : f32
  }
  func.func @d_caller_write_const(%a: memref<f32>, %x: f32, %dr: f32) -> f32 {
    %0 = enzyme.autodiff @caller_write_const(%a, %x, %dr) <{
      activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (memref<f32>, f32, f32) -> f32
    return %0 : f32
  }

  // 5. Distinct memref.alloca buffers passed to two arguments -> no aliasing,
  //    only the written buffer is cloned.
  func.func @caller_distinct_allocs(%x: f32) -> f32 {
    %a = memref.alloca() : memref<f32>
    %b = memref.alloca() : memref<f32>
    memref.store %x, %a[] : memref<f32>
    memref.store %x, %b[] : memref<f32>
    %r = func.call @alias_same(%a, %b, %x) : (memref<f32>, memref<f32>, f32) -> f32
    return %r : f32
  }
  func.func @d_caller_distinct_allocs(%x: f32, %dr: f32) -> f32 {
    %0 = enzyme.autodiff @caller_distinct_allocs(%x, %dr) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (f32, f32) -> f32
    return %0 : f32
  }

  // 6. Writing callee inside scf.for -> hoists the per-iteration clone cache
  //    and materializes a contiguous memref<f32> argument for @diffestep.
  func.func @step(%a: memref<f32>, %x: f32) {
    %v = memref.load %a[] : memref<f32>
    %n = arith.mulf %v, %x : f32
    memref.store %n, %a[] : memref<f32>
    return
  }
  func.func @caller_loop(%a: memref<f32>, %x: f32, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      func.call @step(%a, %x) : (memref<f32>, f32) -> ()
    }
    return
  }
  func.func @d_caller_loop(%a: memref<f32>, %da: memref<f32>, %x: f32, %n: index) -> f32 {
    %0 = enzyme.autodiff @caller_loop(%a, %da, %x, %n) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>],
      ret_activity = []
    }> : (memref<f32>, memref<f32>, f32, index) -> f32
    return %0 : f32
  }
}

// CHECK-LABEL: func.func private @diffecaller_ro_clean(
// CHECK-NOT:     memref.alloc
// CHECK:         call @ro(
// CHECK:         call @differo(

// CHECK-LABEL: func.func private @diffecaller_ro_overwritten(
// CHECK-SAME:    %[[A:.+]]: memref<f32>, %[[DA:.+]]: memref<f32>,
// CHECK:         %[[ALLOC:.+]] = memref.alloc() : memref<f32>
// CHECK-NEXT:    memref.copy %[[A]], %[[ALLOC]]
// CHECK-NEXT:    call @ro(%[[A]],
// CHECK:         call @differo{{(_[0-9]+)?}}(%[[ALLOC]], %[[DA]],
// CHECK-NEXT:    memref.dealloc %[[ALLOC]]

// CHECK-LABEL: func.func private @diffecaller_alias_same(
// CHECK-SAME:    %[[A:.+]]: memref<f32>, %[[DA:.+]]: memref<f32>,
// CHECK:         %[[ALLOC:.+]] = memref.alloc() : memref<f32>
// CHECK-NEXT:    memref.copy %[[A]], %[[ALLOC]]
// CHECK-NEXT:    call @alias_same(%[[A]], %[[A]],
// CHECK-NEXT:    call @diffealias_same{{(_[0-9]+)?}}(%[[ALLOC]], %[[DA]], %[[ALLOC]], %[[DA]],
// CHECK-NEXT:    memref.dealloc %[[ALLOC]]

// CHECK-LABEL: func.func private @diffecaller_write_const(
// CHECK-SAME:    %[[A:.+]]: memref<f32>,
// CHECK:         %[[ALLOC:.+]] = memref.alloc() : memref<f32>
// CHECK-NEXT:    memref.copy %[[A]], %[[ALLOC]]
// CHECK-NEXT:    call @write_const(%[[A]],
// CHECK-NEXT:    call @diffewrite_const(%[[ALLOC]],
// CHECK-NEXT:    memref.dealloc %[[ALLOC]]

// CHECK-LABEL: func.func private @diffecaller_distinct_allocs(
// CHECK:         %[[ALLOC:.+]] = memref.alloc() : memref<f32>
// CHECK-NEXT:    memref.copy %[[A0:[^,]+]], %[[ALLOC]]
// CHECK-NEXT:    call @alias_same(%[[A0]], %[[B0:[^,]+]],
// CHECK-NEXT:    call @diffealias_same{{(_[0-9]+)?}}(%[[ALLOC]], %{{[^,]+}}, %[[B0]], %{{[^,]+}},
// CHECK-NEXT:    memref.dealloc %[[ALLOC]]

// CHECK-LABEL: func.func private @diffecaller_loop(
// CHECK:         %[[CACHE:.+]] = memref.alloc(%{{.+}}) : memref<?xf32>
// CHECK:         scf.for
// CHECK:           %[[SVF:.+]] = memref.subview %[[CACHE]]
// CHECK:           memref.copy %{{.+}}, %[[SVF]]
// CHECK:           func.call @step(
// CHECK:         }
// CHECK:         scf.for
// CHECK:           %[[SVR:.+]] = memref.subview %[[CACHE]]
// CHECK:           %[[TMP:.+]] = memref.alloc() : memref<f32>
// CHECK:           memref.copy %[[SVR]], %[[TMP]]
// CHECK:           func.call @diffestep(%[[TMP]],
// CHECK:           memref.dealloc %[[TMP]]
// CHECK:         }
// CHECK:         memref.dealloc %[[CACHE]]
