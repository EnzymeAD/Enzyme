// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s

// Loading a pointer out of a pointer gives an active value whose derivative is
// a shadow pointer, not a number to accumulate. The load adjoints used to reach
// for createAddOp on it all the same, and a pointer has no addition to give.
// What stands for it is the handle held at the same place in the shadow: the
// same load, off the shadow address.

llvm.func @f_llvm(%pp: !llvm.ptr, %out: !llvm.ptr) {
  %p = llvm.load %pp : !llvm.ptr -> !llvm.ptr
  %v = llvm.load %p : !llvm.ptr -> f64
  %s = arith.mulf %v, %v : f64
  llvm.store %s, %out : f64, !llvm.ptr
  llvm.return
}

func.func @df_llvm(%pp: !llvm.ptr, %dpp: !llvm.ptr, %out: !llvm.ptr, %dout: !llvm.ptr) {
  enzyme.autodiff @f_llvm(%pp, %dpp, %out, %dout) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  return
}

// The shadow pointer is read out of the shadow of %pp, and d(v*v) = 2v lands
// through it. Nothing is added into the pointer itself, and the shadow is a
// real value rather than a placeholder left standing.

// CHECK: llvm.func @diffef_llvm(%[[pp:.+]]: !llvm.ptr, %[[dpp:.+]]: !llvm.ptr, %[[out:.+]]: !llvm.ptr, %[[dout:.+]]: !llvm.ptr)
// CHECK-NOT:     enzyme.placeholder
// CHECK:         %[[dp:.+]] = llvm.load %[[dpp]] : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[p:.+]] = llvm.load %[[pp]] : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[v:.+]] = llvm.load %[[p]] : !llvm.ptr -> f64
// CHECK:         %[[old:.+]] = llvm.load %[[dp]] : !llvm.ptr -> f64
// CHECK:         %[[new:.+]] = arith.addf %[[old]], %{{.+}} fastmath<fast> : f64
// CHECK:         llvm.store %[[new]], %[[dp]] : f64, !llvm.ptr
// CHECK-NOT:     enzyme.placeholder
// CHECK:         llvm.return

// -----

// The same through memref, where the pointer comes out of a memref<?x!llvm.ptr>.

func.func @f_memref(%m: memref<?x!llvm.ptr>, %out: !llvm.ptr) {
  %c0 = arith.constant 0 : index
  %p = memref.load %m[%c0] : memref<?x!llvm.ptr>
  %v = llvm.load %p : !llvm.ptr -> f64
  %s = arith.mulf %v, %v : f64
  llvm.store %s, %out : f64, !llvm.ptr
  return
}

func.func @df_memref(%m: memref<?x!llvm.ptr>, %dm: memref<?x!llvm.ptr>, %out: !llvm.ptr, %dout: !llvm.ptr) {
  enzyme.autodiff @f_memref(%m, %dm, %out, %dout) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity=[] } : (memref<?x!llvm.ptr>, memref<?x!llvm.ptr>, !llvm.ptr, !llvm.ptr) -> ()
  return
}

// CHECK: func.func private @diffef_memref(%[[m:.+]]: memref<?x!llvm.ptr>, %[[dm:.+]]: memref<?x!llvm.ptr>, %[[out2:.+]]: !llvm.ptr, %[[dout2:.+]]: !llvm.ptr)
// CHECK-NOT:     enzyme.placeholder
// CHECK:         %[[dp2:.+]] = memref.load %[[dm]]{{\[}}%{{.+}}] : memref<?x!llvm.ptr>
// CHECK:         %[[p2:.+]] = memref.load %[[m]]{{\[}}%{{.+}}] : memref<?x!llvm.ptr>
// CHECK:         %[[v2:.+]] = llvm.load %[[p2]] : !llvm.ptr -> f64
// CHECK:         %[[old2:.+]] = llvm.load %[[dp2]] : !llvm.ptr -> f64
// CHECK:         %[[new2:.+]] = arith.addf %[[old2]], %{{.+}} fastmath<fast> : f64
// CHECK:         llvm.store %[[new2]], %[[dp2]] : f64, !llvm.ptr
// CHECK-NOT:     enzyme.placeholder
// CHECK:         return

// -----

// And through affine.load, where the map rather than an index operand picks the
// element out.

func.func @f_affine(%m: memref<?x!llvm.ptr>, %out: !llvm.ptr) {
  affine.for %i = 0 to 4 {
    %p = affine.load %m[%i] : memref<?x!llvm.ptr>
    %v = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %out : f64, !llvm.ptr
  }
  return
}

func.func @df_affine(%m: memref<?x!llvm.ptr>, %dm: memref<?x!llvm.ptr>, %out: !llvm.ptr, %dout: !llvm.ptr) {
  enzyme.autodiff @f_affine(%m, %dm, %out, %dout) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity=[] } : (memref<?x!llvm.ptr>, memref<?x!llvm.ptr>, !llvm.ptr, !llvm.ptr) -> ()
  return
}

// The shadow pointer is read out of the shadow memref on the way forward and
// kept, and the reverse loop accumulates through what it kept.

// CHECK: func.func private @diffef_affine(%[[m3:.+]]: memref<?x!llvm.ptr>, %[[dm3:.+]]: memref<?x!llvm.ptr>, %[[out3:.+]]: !llvm.ptr, %[[dout3:.+]]: !llvm.ptr)
// CHECK-NOT:     enzyme.placeholder
// CHECK:         %[[cache:.+]] = memref.alloc() : memref<4x!llvm.ptr>
// CHECK:         affine.for %[[iv:.+]] = 0 to 4 {
// CHECK:           %[[dp3:.+]] = affine.load %[[dm3]]{{\[}}%[[iv]]] : memref<?x!llvm.ptr>
// CHECK:           memref.store %[[dp3]], %[[cache]]{{\[}}%[[iv]]]
// CHECK:           affine.load %[[m3]]{{\[}}%[[iv]]] : memref<?x!llvm.ptr>
// CHECK:         affine.for
// CHECK:           %[[rdp3:.+]] = memref.load %[[cache]]
// CHECK:           %[[old3:.+]] = llvm.load %[[rdp3]] : !llvm.ptr -> f64
// CHECK:           %[[new3:.+]] = arith.addf %[[old3]], %{{.+}} fastmath<fast> : f64
// CHECK:           llvm.store %[[new3]], %[[rdp3]] : f64, !llvm.ptr
// CHECK-NOT:     enzyme.placeholder
// CHECK:         return
