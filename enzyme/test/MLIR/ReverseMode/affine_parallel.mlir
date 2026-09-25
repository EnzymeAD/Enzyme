// RUN: %eopt %s --split-input-file --pass-pipeline="builtin.module(enzyme{dataflow},canonicalize,remove-unnecessary-enzyme-ops,flatten-enzyme-caches,canonicalize)" | FileCheck %s

func.func @foo(%x: memref<?xf32>, %y: memref<?xf32>) {
  affine.parallel (%arg9) = (0) to (4) {
    %0 = memref.load %x[%arg9] : memref<?xf32>
    %1 = arith.mulf %0, %0 : f32
    affine.store %1, %y[%arg9] : memref<?xf32>
  }
  return
}

func.func @dfoo(%x: memref<?xf32>, %dx: memref<?xf32>, %y: memref<?xf32>, %dy: memref<?xf32>) {
  enzyme.autodiff @foo(%x, %dx, %y, %dy) {
    activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity = []
  } : (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  return
}

// CHECK:  func.func @foo(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
// CHECK-NEXT:    affine.parallel (%arg2) = (0) to (4) {
// CHECK-NEXT:      %0 = memref.load %arg0[%arg2] : memref<?xf32>
// CHECK-NEXT:      %1 = arith.mulf %0, %0 : f32
// CHECK-NEXT:      affine.store %1, %arg1[%arg2] : memref<?xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func @dfoo(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
// CHECK-NEXT:    call @diffefoo(%arg0, %arg1, %arg2, %arg3) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @diffefoo(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc() : memref<4xf32>
// CHECK-NEXT:    affine.parallel (%arg4) = (0) to (4) {
// CHECK-NEXT:      %0 = memref.load %arg0[%arg4] : memref<?xf32>
// CHECK-NEXT:      memref.store %0, %alloc[%arg4] : memref<4xf32>
// CHECK-NEXT:      %1 = arith.mulf %0, %0 : f32
// CHECK-NEXT:      affine.store %1, %arg2[%arg4] : memref<?xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.parallel (%arg4) = (0) to (4) {
// CHECK-NEXT:      %0 = memref.load %alloc[%arg4] : memref<4xf32>
// CHECK-NEXT:      %1 = memref.load %arg3[%arg4] : memref<?xf32>
// CHECK-NEXT:      memref.store %cst, %arg3[%arg4] : memref<?xf32>
// CHECK-NEXT:      %2 = arith.mulf %1, %0 fastmath<fast> : f32
// CHECK-NEXT:      %3 = arith.mulf %1, %0 fastmath<fast> : f32
// CHECK-NEXT:      %4 = arith.addf %2, %3 fastmath<fast> : f32
// CHECK-NEXT:      %5 = enzyme.atomic_rmw addf %4, %arg1[%arg4] monotonic fastmath<fast> : (f32, memref<?xf32>) -> f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<4xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----

func.func @nonconst_bound(%x: memref<?xf32>, %y: memref<?xf32>, %len: index) {
  affine.parallel (%arg9) = (0) to (%len) {
    %0 = memref.load %x[%arg9] : memref<?xf32>
    %1 = arith.mulf %0, %0 : f32
    affine.store %1, %y[%arg9] : memref<?xf32>
  }
  return
}

func.func @dnonconst(%x: memref<?xf32>, %dx: memref<?xf32>, %y: memref<?xf32>, %dy: memref<?xf32>) {
  %c4 = arith.constant 4 : index
  enzyme.autodiff @nonconst_bound(%x, %dx, %y, %dy, %c4) {
    activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>],
    ret_activity = []
  } : (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, index) -> ()
  return
}

// CHECK:  func.func @nonconst_bound(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: index) {
// CHECK-NEXT:    affine.parallel (%arg3) = (0) to (symbol(%arg2)) {
// CHECK-NEXT:      %0 = memref.load %arg0[%arg3] : memref<?xf32>
// CHECK-NEXT:      %1 = arith.mulf %0, %0 : f32
// CHECK-NEXT:      affine.store %1, %arg1[%arg3] : memref<?xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func @dnonconst(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    call @diffenonconst_bound(%arg0, %arg1, %arg2, %arg3, %c4) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, index) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @diffenonconst_bound(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: index) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc(%arg4) : memref<?xf32>
// CHECK-NEXT:    affine.parallel (%arg5) = (0) to (symbol(%arg4)) {
// CHECK-NEXT:      %0 = memref.load %arg0[%arg5] : memref<?xf32>
// CHECK-NEXT:      memref.store %0, %alloc[%arg5] : memref<?xf32>
// CHECK-NEXT:      %1 = arith.mulf %0, %0 : f32
// CHECK-NEXT:      affine.store %1, %arg2[%arg5] : memref<?xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.parallel (%arg5) = (0) to (symbol(%arg4)) {
// CHECK-NEXT:      %0 = memref.load %alloc[%arg5] : memref<?xf32>
// CHECK-NEXT:      %1 = memref.load %arg3[%arg5] : memref<?xf32>
// CHECK-NEXT:      memref.store %cst, %arg3[%arg5] : memref<?xf32>
// CHECK-NEXT:      %2 = arith.mulf %1, %0 fastmath<fast> : f32
// CHECK-NEXT:      %3 = arith.mulf %1, %0 fastmath<fast> : f32
// CHECK-NEXT:      %4 = arith.addf %2, %3 fastmath<fast> : f32
// CHECK-NEXT:      %5 = enzyme.atomic_rmw addf %4, %arg1[%arg5] monotonic fastmath<fast> : (f32, memref<?xf32>) -> f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<?xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----

func.func @non_1_step(%x: memref<?xf32>, %y: memref<?xf32>) {
  affine.parallel (%arg9) = (0) to (4) step (2) {
    %0 = memref.load %x[%arg9] : memref<?xf32>
    %1 = arith.mulf %0, %0 : f32
    affine.store %1, %y[%arg9] : memref<?xf32>
  }
  return
}

func.func @dnon_1_step(%x: memref<?xf32>, %dx: memref<?xf32>, %y: memref<?xf32>, %dy: memref<?xf32>) {
  enzyme.autodiff @non_1_step(%x, %dx, %y, %dy) {
    activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity = []
  } : (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  return
}

// CHECK:  func.func @non_1_step(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
// CHECK-NEXT:    affine.parallel (%arg2) = (0) to (4) step (2) {
// CHECK-NEXT:      %0 = memref.load %arg0[%arg2] : memref<?xf32>
// CHECK-NEXT:      %1 = arith.mulf %0, %0 : f32
// CHECK-NEXT:      affine.store %1, %arg1[%arg2] : memref<?xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func @dnon_1_step(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
// CHECK-NEXT:    call @diffenon_1_step(%arg0, %arg1, %arg2, %arg3) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @diffenon_1_step(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc() : memref<2xf32>
// CHECK-NEXT:    affine.parallel (%arg4) = (0) to (4) step (2) {
// CHECK-NEXT:      %0 = arith.divui %arg4, %c2 : index
// CHECK-NEXT:      %1 = memref.load %arg0[%arg4] : memref<?xf32>
// CHECK-NEXT:      memref.store %1, %alloc[%0] : memref<2xf32>
// CHECK-NEXT:      %2 = arith.mulf %1, %1 : f32
// CHECK-NEXT:      affine.store %2, %arg2[%arg4] : memref<?xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.parallel (%arg4) = (0) to (4) step (2) {
// CHECK-NEXT:      %0 = arith.divui %arg4, %c2 : index
// CHECK-NEXT:      %1 = memref.load %alloc[%0] : memref<2xf32>
// CHECK-NEXT:      %2 = memref.load %arg3[%arg4] : memref<?xf32>
// CHECK-NEXT:      memref.store %cst, %arg3[%arg4] : memref<?xf32>
// CHECK-NEXT:      %3 = arith.mulf %2, %1 fastmath<fast> : f32
// CHECK-NEXT:      %4 = arith.mulf %2, %1 fastmath<fast> : f32
// CHECK-NEXT:      %5 = arith.addf %3, %4 fastmath<fast> : f32
// CHECK-NEXT:      %6 = enzyme.atomic_rmw addf %5, %arg1[%arg4] monotonic fastmath<fast> : (f32, memref<?xf32>) -> f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<2xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----

func.func @par2d(%x: memref<3x3xf32>, %y: memref<3x3xf32>) {
  affine.parallel (%iv, %jv) = (0, 0) to (3, 3) {
    %0 = affine.load %x[%iv, %jv] : memref<3x3xf32>
    %1 = arith.mulf %0, %0 : f32
    affine.store %1, %y[%iv, %jv] : memref<3x3xf32>
  }
  return
}

func.func @dpar2d(%x: memref<3x3xf32>, %dx: memref<3x3xf32>, %y: memref<3x3xf32>, %dy: memref<3x3xf32>) {
  enzyme.autodiff @par2d(%x, %dx, %y, %dy) {
    activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity = []
  } : (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// CHECK:  func.func @par2d(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
// CHECK-NEXT:    affine.parallel (%arg2, %arg3) = (0, 0) to (3, 3) {
// CHECK-NEXT:      %0 = affine.load %arg0[%arg2, %arg3] : memref<3x3xf32>
// CHECK-NEXT:      %1 = arith.mulf %0, %0 : f32
// CHECK-NEXT:      affine.store %1, %arg1[%arg2, %arg3] : memref<3x3xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func @dpar2d(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>, %arg3: memref<3x3xf32>) {
// CHECK-NEXT:    call @diffepar2d(%arg0, %arg1, %arg2, %arg3) : (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @diffepar2d(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>, %arg3: memref<3x3xf32>) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc() : memref<3x3xf32>
// CHECK-NEXT:    affine.parallel (%arg4, %arg5) = (0, 0) to (3, 3) {
// CHECK-NEXT:      %0 = affine.load %arg0[%arg4, %arg5] : memref<3x3xf32>
// CHECK-NEXT:      memref.store %0, %alloc[%arg4, %arg5] : memref<3x3xf32>
// CHECK-NEXT:      %1 = arith.mulf %0, %0 : f32
// CHECK-NEXT:      affine.store %1, %arg2[%arg4, %arg5] : memref<3x3xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.parallel (%arg4, %arg5) = (0, 0) to (3, 3) {
// CHECK-NEXT:      %0 = memref.load %alloc[%arg4, %arg5] : memref<3x3xf32>
// CHECK-NEXT:      %1 = memref.load %arg3[%arg4, %arg5] : memref<3x3xf32>
// CHECK-NEXT:      memref.store %cst, %arg3[%arg4, %arg5] : memref<3x3xf32>
// CHECK-NEXT:      %2 = arith.mulf %1, %0 fastmath<fast> : f32
// CHECK-NEXT:      %3 = arith.mulf %1, %0 fastmath<fast> : f32
// CHECK-NEXT:      %4 = arith.addf %2, %3 fastmath<fast> : f32
// CHECK-NEXT:      %5 = enzyme.atomic_rmw addf %4, %arg1[%arg4, %arg5] monotonic fastmath<fast> : (f32, memref<3x3xf32>) -> f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<3x3xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----

// verify that the dataflow activity analysis runs on affine.parallel
func.func @some_inactive(%x: memref<3x3xf32>, %inactive: memref<3x3xf32>, %y: memref<3x3xf32>) {
  affine.parallel (%iv, %jv) = (0, 0) to (3, 3) {
    %0 = affine.load %x[%iv, %jv] : memref<3x3xf32>
    %i0 = affine.load %inactive[%iv, %jv] : memref<3x3xf32>
    %i1 = math.exp %i0 : f32
    %i2 = math.sin %i1 : f32
    %i3 = math.tanh %i2 : f32
    %i4 = arith.mulf %i3, %i3 : f32
    %1 = arith.mulf %i4, %0 : f32
    affine.store %1, %y[%iv, %jv] : memref<3x3xf32>
  }
  return
}

func.func @dsome_inactive(%x: memref<3x3xf32>, %dx: memref<3x3xf32>, %inactive: memref<3x3xf32>, %y: memref<3x3xf32>, %dy: memref<3x3xf32>) {
  enzyme.autodiff @some_inactive(%x, %dx, %inactive, %y, %dy) {
    activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>],
    ret_activity = []
  } : (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// CHECK:  func.func @some_inactive(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
// CHECK-NEXT:    affine.parallel (%arg3, %arg4) = (0, 0) to (3, 3) {
// CHECK-NEXT:      %0 = affine.load %arg0[%arg3, %arg4] : memref<3x3xf32>
// CHECK-NEXT:      %1 = affine.load %arg1[%arg3, %arg4] : memref<3x3xf32>
// CHECK-NEXT:      %2 = math.exp %1 : f32
// CHECK-NEXT:      %3 = math.sin %2 : f32
// CHECK-NEXT:      %4 = math.tanh %3 : f32
// CHECK-NEXT:      %5 = arith.mulf %4, %4 : f32
// CHECK-NEXT:      %6 = arith.mulf %5, %0 : f32
// CHECK-NEXT:      affine.store %6, %arg2[%arg3, %arg4] : memref<3x3xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func @dsome_inactive(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>, %arg3: memref<3x3xf32>, %arg4: memref<3x3xf32>) {
// CHECK-NEXT:    call @diffesome_inactive(%arg0, %arg1, %arg2, %arg3, %arg4) : (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @diffesome_inactive(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>, %arg3: memref<3x3xf32>, %arg4: memref<3x3xf32>) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc() : memref<3x3xf32>
// CHECK-NEXT:    affine.parallel (%arg5, %arg6) = (0, 0) to (3, 3) {
// CHECK-NEXT:      %0 = affine.load %arg0[%arg5, %arg6] : memref<3x3xf32>
// CHECK-NEXT:      %1 = affine.load %arg2[%arg5, %arg6] : memref<3x3xf32>
// CHECK-NEXT:      %2 = math.exp %1 : f32
// CHECK-NEXT:      %3 = math.sin %2 : f32
// CHECK-NEXT:      %4 = math.tanh %3 : f32
// CHECK-NEXT:      %5 = arith.mulf %4, %4 : f32
// CHECK-NEXT:      memref.store %5, %alloc[%arg5, %arg6] : memref<3x3xf32>
// CHECK-NEXT:      %6 = arith.mulf %5, %0 : f32
// CHECK-NEXT:      affine.store %6, %arg3[%arg5, %arg6] : memref<3x3xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.parallel (%arg5, %arg6) = (0, 0) to (3, 3) {
// CHECK-NEXT:      %0 = memref.load %alloc[%arg5, %arg6] : memref<3x3xf32>
// CHECK-NEXT:      %1 = memref.load %arg4[%arg5, %arg6] : memref<3x3xf32>
// CHECK-NEXT:      memref.store %cst, %arg4[%arg5, %arg6] : memref<3x3xf32>
// CHECK-NEXT:      %2 = arith.mulf %1, %0 fastmath<fast> : f32
// CHECK-NEXT:      %3 = enzyme.atomic_rmw addf %2, %arg1[%arg5, %arg6] monotonic fastmath<fast> : (f32, memref<3x3xf32>) -> f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<3x3xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
