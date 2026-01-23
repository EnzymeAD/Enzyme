// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --lower-llvm-ext --canonicalize %s | FileCheck %s

module {
  llvm.func @square(%x: f64) -> f64 {
    %next = arith.mulf %x, %x : f64
    llvm.return %next : f64
  }

  func.func @dsquare(%x: f64, %dr: f64) -> f64 {
    %dx = enzyme.autodiff @square(%x, %dr)
      {
        activity=[#enzyme<activity enzyme_active>],
        ret_activity=[#enzyme<activity enzyme_activenoneed>]
      } : (f64, f64) -> f64
    return %dx : f64
  }

// CHECK: llvm.func @diffesquare(%arg0: f64, %arg1: f64) -> f64 attributes {sym_visibility = "private"} {
// CHECK-NEXT:    %0 = arith.mulf %arg1, %arg0 : f64
// CHECK-NEXT:    %1 = arith.mulf %arg1, %arg0 : f64
// CHECK-NEXT:    %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:    llvm.return %2 : f64
// CHECK-NEXT:  }
}

// -----

module {
llvm.func @multireturn(%x: f64, %y: f64) -> f64 {
  %0 = arith.mulf %x, %y : f64
  llvm.return %0 : f64
}

func.func @dmultireturn(%x: f64, %y: f64, %dr: f64) -> (f64, f64) {
  %res = enzyme.autodiff @multireturn(%x, %y, %dr)
    {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64, f64) -> !llvm.struct<(f64, f64)>
  %fst = llvm.extractvalue %res[0] : !llvm.struct<(f64, f64)>
  %snd = llvm.extractvalue %res[1] : !llvm.struct<(f64, f64)>
  return %fst, %snd : f64, f64
}
}

// CHECK: llvm.func @diffemultireturn(%arg0: f64, %arg1: f64, %arg2: f64) -> !llvm.struct<(f64, f64)> attributes {sym_visibility = "private"} {
// CHECK-NEXT:    %0 = llvm.mlir.poison : !llvm.struct<(f64, f64)>
// CHECK-NEXT:    %1 = arith.mulf %arg2, %arg1 : f64
// CHECK-NEXT:    %2 = arith.mulf %arg2, %arg0 : f64
// CHECK-NEXT:    %3 = llvm.insertvalue %1, %0[0] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:    %4 = llvm.insertvalue %2, %3[1] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:    llvm.return %4 : !llvm.struct<(f64, f64)>
// CHECK-NEXT:  }

// -----

module {
llvm.func @loadstore(%a: !llvm.ptr, %b: f32) -> f32 {
  %sz = arith.constant 32 : i64
  llvm_ext.ptr_size_hint %a, %sz : !llvm.ptr, i64
  llvm.store %b, %a : f32, !llvm.ptr
  %0 = llvm.load %a : !llvm.ptr -> f32
  llvm.return %0 : f32
}

func.func @dloadstore(%a: !llvm.ptr, %da: !llvm.ptr, %b: f32, %dres: f32) -> f32 {
  %res = enzyme.autodiff @loadstore(%a, %da, %b, %dres)
    {
      activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (!llvm.ptr, !llvm.ptr, f32, f32) -> f32
  return %res : f32
}
}

// CHECK:  llvm.func @diffeloadstore(%[[a:.+]]: !llvm.ptr, %[[da:.+]]: !llvm.ptr, %[[b:.+]]: f32, %[[dres:.+]]: f32) -> f32 attributes {sym_visibility = "private"} {
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    llvm.store %[[b]], %[[a]] : f32, !llvm.ptr
// CHECK-NEXT:    %[[daval1:.+]] = llvm.load %[[da]] : !llvm.ptr -> f32
// CHECK-NEXT:    %[[daval2:.+]] = arith.addf %[[daval1]], %[[dres]] : f32
// CHECK-NEXT:    llvm.store %[[daval2]], %[[da]] : f32, !llvm.ptr
// CHECK-NEXT:    %[[daval3:.+]] = llvm.load %[[da]] : !llvm.ptr -> f32
// CHECK-NEXT:    llvm.store %[[zero]], %[[da]] : f32, !llvm.ptr
// CHECK-NEXT:    llvm.return %[[daval3]] : f32
// CHECK-NEXT:  }

// -----

module {
llvm.func @f_iter(%a: !llvm.ptr) -> f32 {
  %lb = arith.constant 0 : index
  %ub = arith.constant 9 : index
  %step = arith.constant 1 : index

  %prod_0 = arith.constant 0.00 : f32

  %prod = scf.for %iv = %lb to %ub step %step
      iter_args(%prod_iter = %prod_0) -> f32 {
    %i = arith.index_cast %iv : index to i32
    %ptr = llvm.getelementptr %a[%i] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %val = llvm.load %ptr : !llvm.ptr -> f32
    %prod_next = arith.mulf %val, %prod_iter : f32
    scf.yield %prod_next : f32
  }

  llvm.return %prod : f32
}
func.func @f_iter_autodiff(%a: !llvm.ptr, %da: !llvm.ptr, %dres: f32) {
  enzyme.autodiff @f_iter(%a, %da, %dres)
    {
      activity=[#enzyme<activity enzyme_dup>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (!llvm.ptr, !llvm.ptr, f32) -> ()
  return
}
}

// CHECK:  llvm.func @diffef_iter(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: f32) attributes {sym_visibility = "private"} {
// CHECK-NEXT:    %c8 = arith.constant 8 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc() : memref<9xf32>
// CHECK-NEXT:    %alloc_0 = memref.alloc() : memref<9xf32>
// CHECK-NEXT:    %0 = scf.for %arg3 = %c0 to %c9 step %c1 iter_args(%arg4 = %cst) -> (f32) {
// CHECK-NEXT:      memref.store %arg4, %alloc_0[%arg3] : memref<9xf32>
// CHECK-NEXT:      %2 = arith.index_cast %arg3 : index to i32
// CHECK-NEXT:      %3 = llvm.getelementptr %arg0[%2] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// CHECK-NEXT:      %4 = llvm.load %3 : !llvm.ptr -> f32
// CHECK-NEXT:      memref.store %4, %alloc[%arg3] : memref<9xf32>
// CHECK-NEXT:      %5 = arith.mulf %4, %arg4 : f32
// CHECK-NEXT:      scf.yield %5 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = scf.for %arg3 = %c0 to %c9 step %c1 iter_args(%arg4 = %arg2) -> (f32) {
// CHECK-NEXT:      %[[ridx:.+]] = arith.subi %c8, %arg3 : index
// CHECK-NEXT:      %3 = memref.load %alloc[%2] : memref<9xf32>
// CHECK-NEXT:      %4 = memref.load %alloc_0[%2] : memref<9xf32>
// CHECK-NEXT:      %5 = arith.index_cast %[[ridx]] : index to i32
// CHECK-NEXT:      %6 = llvm.getelementptr %arg1[%5] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// CHECK-NEXT:      %7 = arith.mulf %arg4, %4 : f32
// CHECK-NEXT:      %8 = arith.mulf %arg4, %3 : f32
// CHECK-NEXT:      %9 = llvm.load %6 : !llvm.ptr -> f32
// CHECK-NEXT:      %10 = arith.addf %9, %7 : f32
// CHECK-NEXT:      llvm.store %10, %6 : f32, !llvm.ptr
// CHECK-NEXT:      scf.yield %8 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc_0 : memref<9xf32>
// CHECK-NEXT:    memref.dealloc %alloc : memref<9xf32>
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
