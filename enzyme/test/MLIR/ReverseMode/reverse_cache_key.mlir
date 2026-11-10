// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --cse %s | FileCheck %s

module {
  func.func @square(%x: f64) -> f64 {
    %next = arith.mulf %x, %x : f64
    return %next : f64
  }

  func.func @weak_and_strong(%x: f64, %dr: f64) -> (f64, f64, f64, f64, f64) {
    %dx0 = enzyme.autodiff @square(%x, %dr) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>],
      strong_zero = true
    }> : (f64, f64) -> f64
    %p, %dx1 = enzyme.autodiff @square(%x, %dr) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (f64, f64) -> (f64, f64)
    %dx2 = enzyme.autodiff @square(%x, %dr) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (f64, f64) -> f64
    %dx3 = enzyme.autodiff @square(%x, %dr) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>],
      strong_zero = true
    }> : (f64, f64) -> f64
    return %p, %dx0, %dx1, %dx2, %dx3 : f64, f64, f64, f64, f64
  }
}

// CHECK-LABEL: func.func @weak_and_strong(
// CHECK-SAME:    %[[X:.+]]: f64, %[[DR:.+]]: f64) -> (f64, f64, f64, f64, f64) {
// CHECK-NEXT:    %[[DX0:.+]] = call @diffesquare(%[[X]], %[[DR]]) : (f64, f64) -> f64
// CHECK-NEXT:    %[[ACT:.+]]:2 = call @diffesquare_0(%[[X]], %[[DR]]) : (f64, f64) -> (f64, f64)
// CHECK-NEXT:    %[[DX2:.+]] = call @diffesquare_1(%[[X]], %[[DR]]) : (f64, f64) -> f64
// CHECK-NEXT:    %[[DX3:.+]] = call @diffesquare(%[[X]], %[[DR]]) : (f64, f64) -> f64
// CHECK-NEXT:    return %[[ACT]]#0, %[[DX0]], %[[ACT]]#1, %[[DX2]], %[[DX3]] : f64, f64, f64, f64, f64
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @diffesquare(
// CHECK-SAME:    %[[S0:.+]]: f64, %[[S1:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[ISZ:.+]] = arith.cmpf oeq, %[[S1]], %[[ZERO]] fastmath<fast> : f64
// CHECK-NEXT:    %[[PROD:.+]] = arith.mulf %[[S1]], %[[S0]] fastmath<fast> : f64
// CHECK-NEXT:    %[[SEL:.+]] = arith.select %[[ISZ]], %[[ZERO]], %[[PROD]] : f64
// CHECK-NEXT:    %[[SUM:.+]] = arith.addf %[[SEL]], %[[SEL]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[SUM]] : f64
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @diffesquare_0(
// CHECK-SAME:    %[[A0:.+]]: f64, %[[A1:.+]]: f64) -> (f64, f64) {
// CHECK-NEXT:    %[[PRIMAL:.+]] = arith.mulf %[[A0]], %[[A0]] : f64
// CHECK-NEXT:    %[[APROD:.+]] = arith.mulf %[[A1]], %[[A0]] fastmath<fast> : f64
// CHECK-NEXT:    %[[ASUM:.+]] = arith.addf %[[APROD]], %[[APROD]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[PRIMAL]], %[[ASUM]] : f64, f64
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @diffesquare_1(
// CHECK-SAME:    %[[W0:.+]]: f64, %[[W1:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[WPROD:.+]] = arith.mulf %[[W1]], %[[W0]] fastmath<fast> : f64
// CHECK-NEXT:    %[[WSUM:.+]] = arith.addf %[[WPROD]], %[[WPROD]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[WSUM]] : f64
// CHECK-NEXT:  }
// CHECK-NOT:   func.func private @diffesquare_2

// -----

module {
  func.func @load_elem(%mem: memref<4xf32>, %idx: index) -> f32 {
    %val = memref.load %mem[%idx] : memref<4xf32>
    return %val : f32
  }

  func.func @dload_elem(%mem: memref<4xf32>, %dmem: memref<4xf32>, %idx: index, %dout: f32) {
    enzyme.autodiff @load_elem(%mem, %dmem, %idx, %dout) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (memref<4xf32>, memref<4xf32>, index, f32) -> ()
    enzyme.autodiff @load_elem(%mem, %dmem, %idx, %dout) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>],
      atomic_add = true
    }> : (memref<4xf32>, memref<4xf32>, index, f32) -> ()
    return
  }
}

// CHECK-LABEL: func.func @dload_elem(
// CHECK-SAME:    %[[MEM:.+]]: memref<4xf32>, %[[DMEM:.+]]: memref<4xf32>, %[[IDX:.+]]: index, %[[DOUT:.+]]: f32) {
// CHECK-NEXT:    call @diffeload_elem(%[[MEM]], %[[DMEM]], %[[IDX]], %[[DOUT]]) : (memref<4xf32>, memref<4xf32>, index, f32) -> ()
// CHECK-NEXT:    call @diffeload_elem_0(%[[MEM]], %[[DMEM]], %[[IDX]], %[[DOUT]]) : (memref<4xf32>, memref<4xf32>, index, f32) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @diffeload_elem(
// CHECK-SAME:    %{{.+}}: memref<4xf32>, %[[LDMEM:.+]]: memref<4xf32>, %[[LIDX:.+]]: index, %[[LDOUT:.+]]: f32) {
// CHECK-NEXT:    %[[PREV:.+]] = memref.load %[[LDMEM]][%[[LIDX]]] : memref<4xf32>
// CHECK-NEXT:    %[[ADDED:.+]] = arith.addf %[[PREV]], %[[LDOUT]] fastmath<fast> : f32
// CHECK-NEXT:    memref.store %[[ADDED]], %[[LDMEM]][%[[LIDX]]] : memref<4xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @diffeload_elem_0(
// CHECK-SAME:    %{{.+}}: memref<4xf32>, %[[ADMEM:.+]]: memref<4xf32>, %[[AIDX:.+]]: index, %[[ADOUT:.+]]: f32) {
// CHECK:         enzyme.atomic_rmw addf %[[ADOUT]], %[[ADMEM]][%[[AIDX]]] monotonic fastmath<fast> : (f32, memref<4xf32>) -> f32
// CHECK-NEXT:    return
// CHECK-NEXT:  }
