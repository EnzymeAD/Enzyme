// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s

// Loading a pointer out of a pointer gives an active value whose derivative is
// a shadow pointer, not a number to accumulate. The load adjoints used to reach
// for createAddOp on it all the same, and a pointer has no addition to give.

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

// The pointer load is replayed and the gradient goes to the shadow it stands
// for; nothing is added into the pointer itself.

// CHECK: llvm.func @diffef_llvm(%[[pp:.+]]: !llvm.ptr, %[[dpp:.+]]: !llvm.ptr, %[[out:.+]]: !llvm.ptr, %[[dout:.+]]: !llvm.ptr)
// CHECK:         %[[shadow:.+]] = "enzyme.placeholder"() : () -> !llvm.ptr
// CHECK:         %[[p:.+]] = llvm.load %[[pp]] : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[v:.+]] = llvm.load %[[p]] : !llvm.ptr -> f64
// CHECK:         %[[old:.+]] = llvm.load %[[shadow]] : !llvm.ptr -> f64
// CHECK:         %[[new:.+]] = arith.addf %[[old]], %{{.+}} fastmath<fast> : f64
// CHECK:         llvm.store %[[new]], %[[shadow]] : f64, !llvm.ptr
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

// CHECK: func.func private @diffef_memref
// CHECK:         memref.load
// CHECK:         return
