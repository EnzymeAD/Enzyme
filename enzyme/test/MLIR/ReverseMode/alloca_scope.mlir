// RUN: %eopt %s --split-input-file --pass-pipeline="builtin.module(enzyme,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math)" | FileCheck %s

func.func @alloca_scope_rev(%x: f64) -> f64 {
  %res = memref.alloca_scope -> (f64) {
    %sq = arith.mulf %x, %x : f64
    memref.alloca_scope.return %sq : f64
  }
  return %res : f64
}

func.func @dalloca_scope_rev(%x: f64, %dres: f64) -> f64 {
  %r = enzyme.autodiff @alloca_scope_rev(%x, %dres) <{
    activity = [#enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_activenoneed>]
  }> : (f64, f64) -> f64
  return %r : f64
}

// CHECK-LABEL: func.func private @diffealloca_scope_rev(
// CHECK-SAME: %[[X:.+]]: f64, %[[DRES:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[M0:.+]] = arith.mulf %[[DRES]], %[[X]] fastmath<fast> : f64
// CHECK-NEXT:    %[[M1:.+]] = arith.mulf %[[DRES]], %[[X]] fastmath<fast> : f64
// CHECK-NEXT:    %[[ADD:.+]] = arith.addf %[[M0]], %[[M1]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[ADD]] : f64
// CHECK-NEXT:  }

// -----

func.func @alloca_scope_memref(%x: memref<f32>, %y: memref<f32>) {
  memref.alloca_scope {
    %val = memref.load %x[] : memref<f32>
    %sq = arith.mulf %val, %val : f32
    memref.store %sq, %y[] : memref<f32>
  }
  return
}

func.func @dalloca_scope_memref(%x: memref<f32>, %dx: memref<f32>, %y: memref<f32>, %dy: memref<f32>) {
  enzyme.autodiff @alloca_scope_memref(%x, %dx, %y, %dy) <{
    activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity = []
  }> : (memref<f32>, memref<f32>, memref<f32>, memref<f32>) -> ()
  return
}

// CHECK-LABEL: func.func private @diffealloca_scope_memref(
// CHECK-SAME: %[[X:.+]]: memref<f32>, %[[DX:.+]]: memref<f32>, %[[Y:.+]]: memref<f32>, %[[DY:.+]]: memref<f32>) {
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[LD:.+]] = memref.load %[[X]][] : memref<f32>
// CHECK-NEXT:    %[[SQ:.+]] = arith.mulf %[[LD]], %[[LD]] : f32
// CHECK-NEXT:    memref.store %[[SQ]], %[[Y]][] : memref<f32>
// CHECK-NEXT:    %[[LDY:.+]] = memref.load %[[DY]][] : memref<f32>
// CHECK-NEXT:    memref.store %[[ZERO]], %[[DY]][] : memref<f32>
// CHECK-NEXT:    %[[M0:.+]] = arith.mulf %[[LDY]], %[[LD]] fastmath<fast> : f32
// CHECK-NEXT:    %[[M1:.+]] = arith.mulf %[[LDY]], %[[LD]] fastmath<fast> : f32
// CHECK-NEXT:    %[[ADD0:.+]] = arith.addf %[[M0]], %[[M1]] fastmath<fast> : f32
// CHECK-NEXT:    %[[LDX:.+]] = memref.load %[[DX]][] : memref<f32>
// CHECK-NEXT:    %[[ADD1:.+]] = arith.addf %[[LDX]], %[[ADD0]] fastmath<fast> : f32
// CHECK-NEXT:    memref.store %[[ADD1]], %[[DX]][] : memref<f32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----

func.func @alloca_scope_with_alloca(%x: f64, %n: index) -> f64 {
  %c0 = arith.constant 0 : index
  %res = memref.alloca_scope -> (f64) {
    %tmp = memref.alloca(%n) : memref<?xf64>
    %sq = arith.mulf %x, %x : f64
    memref.store %sq, %tmp[%c0] : memref<?xf64>
    %ld = memref.load %tmp[%c0] : memref<?xf64>
    memref.alloca_scope.return %ld : f64
  }
  return %res : f64
}

func.func @dalloca_scope_with_alloca(%x: f64, %n: index, %dres: f64) -> f64 {
  %r = enzyme.autodiff @alloca_scope_with_alloca(%x, %n, %dres) <{
    activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>],
    ret_activity = [#enzyme.activity<enzyme_activenoneed>]
  }> : (f64, index, f64) -> f64
  return %r : f64
}

// CHECK-LABEL: func.func private @diffealloca_scope_with_alloca(
// CHECK-SAME: %[[X:.+]]: f64, %[[N:.+]]: index, %[[DRES:.+]]: f64) -> f64 {
// CHECK:         %[[FWD:.+]]:2 = memref.alloca_scope -> (f64, memref<?xf64>) {
// CHECK-NEXT:      %[[DALLOCA:.+]] = memref.alloca(%[[N]]) : memref<?xf64>
// CHECK-NEXT:      enzyme.fill_zero %[[DALLOCA]] : memref<?xf64>
// CHECK-NEXT:      %[[ALLOCA:.+]] = memref.alloca(%[[N]]) : memref<?xf64>
// CHECK-NEXT:      %[[SQ:.+]] = arith.mulf %[[X]], %[[X]] : f64
// CHECK-NEXT:      memref.store %[[SQ]], %[[ALLOCA]][%{{.+}}] : memref<?xf64>
// CHECK-NEXT:      %[[LD:.+]] = memref.load %[[ALLOCA]][%{{.+}}] : memref<?xf64>
// CHECK-NEXT:      memref.alloca_scope.return %[[LD]], %[[DALLOCA]] : f64, memref<?xf64>
// CHECK-NEXT:    }
// CHECK:         return %{{.+}} : f64
// CHECK-NEXT:  }
