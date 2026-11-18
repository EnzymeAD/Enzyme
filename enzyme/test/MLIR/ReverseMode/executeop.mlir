// RUN: %eopt %s --split-input-file --pass-pipeline="builtin.module(enzyme,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math)" | FileCheck %s

func.func @exec_region_rev(%x: f64) -> f64 {
  %res = scf.execute_region -> (f64) no_inline {
    %sq = arith.mulf %x, %x : f64
    scf.yield %sq : f64
  }
  return %res : f64
}

func.func @dexec_region_rev(%x: f64, %dres: f64) -> f64 {
  %r = enzyme.autodiff @exec_region_rev(%x, %dres) <{
    activity = [#enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_activenoneed>]
  }> : (f64, f64) -> f64
  return %r : f64
}

// CHECK-LABEL: func.func private @diffeexec_region_rev(
// CHECK-SAME: %[[X:.+]]: f64, %[[DRES:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[REV:.+]] = scf.execute_region -> f64 no_inline {
// CHECK-NEXT:      %[[M0:.+]] = arith.mulf %[[DRES]], %[[X]] fastmath<fast> : f64
// CHECK-NEXT:      %[[M1:.+]] = arith.mulf %[[DRES]], %[[X]] fastmath<fast> : f64
// CHECK-NEXT:      %[[ADD:.+]] = arith.addf %[[M0]], %[[M1]] fastmath<fast> : f64
// CHECK-NEXT:      scf.yield %[[ADD]] : f64
// CHECK:         return %[[REV]] : f64

// -----

func.func @exec_region_mincut(%x: memref<f32>) {
  scf.execute_region no_inline {
    %val = memref.load %x[] : memref<f32>
    %sin = math.sin %val : f32
    %cos = math.cos %val : f32
    %mul = arith.mulf %sin, %cos : f32
    memref.store %mul, %x[] : memref<f32>
    scf.yield
  }
  return
}

func.func @dexec_region_mincut(%x: memref<f32>, %dx: memref<f32>) {
  enzyme.autodiff @exec_region_mincut(%x, %dx) <{
    activity = [#enzyme.activity<enzyme_dup>],
    ret_activity = []
  }> : (memref<f32>, memref<f32>) -> ()
  return
}

// CHECK-LABEL: func.func private @diffeexec_region_mincut(
// CHECK-SAME: %[[X:.+]]: memref<f32>, %[[DX:.+]]: memref<f32>) {
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[FWD:.+]] = scf.execute_region -> f32 no_inline {
// CHECK-NEXT:      %[[VAL:.+]] = memref.load %[[X]][] : memref<f32>
// CHECK-NEXT:      %[[SIN:.+]] = math.sin %[[VAL]] : f32
// CHECK-NEXT:      %[[COS:.+]] = math.cos %[[VAL]] : f32
// CHECK-NEXT:      %[[MUL:.+]] = arith.mulf %[[SIN]], %[[COS]] : f32
// CHECK-NEXT:      memref.store %[[MUL]], %[[X]][] : memref<f32>
// CHECK-NEXT:      scf.yield %[[VAL]] : f32
// CHECK:         scf.execute_region no_inline {
// CHECK-NEXT:      %[[SIN1:.+]] = math.sin %[[FWD]] : f32
// CHECK-NEXT:      %[[COS1:.+]] = math.cos %[[FWD]] : f32
// CHECK-NEXT:      %[[DY:.+]] = memref.load %[[DX]][] : memref<f32>
// CHECK-NEXT:      memref.store %[[ZERO]], %[[DX]][] : memref<f32>
// CHECK-NEXT:      %[[M0:.+]] = arith.mulf %[[DY]], %[[COS1]] fastmath<fast> : f32
// CHECK-NEXT:      %[[M1:.+]] = arith.mulf %[[DY]], %[[SIN1]] fastmath<fast> : f32
// CHECK-NEXT:      %[[SIN2:.+]] = math.sin %[[FWD]] fastmath<fast> : f32
// CHECK-NEXT:      %[[NEG:.+]] = arith.negf %[[SIN2]] fastmath<fast> : f32
// CHECK-NEXT:      %[[M2:.+]] = arith.mulf %[[M1]], %[[NEG]] fastmath<fast> : f32
// CHECK-NEXT:      %[[COS2:.+]] = math.cos %[[FWD]] fastmath<fast> : f32
// CHECK-NEXT:      %[[M3:.+]] = arith.mulf %[[M0]], %[[COS2]] fastmath<fast> : f32
// CHECK-NEXT:      %[[ADD0:.+]] = arith.addf %[[M2]], %[[M3]] fastmath<fast> : f32
// CHECK-NEXT:      %[[LDX:.+]] = memref.load %[[DX]][] : memref<f32>
// CHECK-NEXT:      %[[ADD1:.+]] = arith.addf %[[LDX]], %[[ADD0]] fastmath<fast> : f32
// CHECK-NEXT:      memref.store %[[ADD1]], %[[DX]][] : memref<f32>
// CHECK-NEXT:      scf.yield
// CHECK:         return

// -----

func.func @exec_region_ptr(%x: !llvm.ptr) -> f32 {
  %c4 = llvm.mlir.constant (4 : i64) : i64
  %ptr = scf.execute_region -> (!llvm.ptr) no_inline {
    %gep = llvm.getelementptr %x[%c4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %val = llvm.load %x : !llvm.ptr -> f32
    %cos = math.cos %val : f32
    llvm.store %cos, %x : f32, !llvm.ptr
    scf.yield %gep : !llvm.ptr
  }
  %ld = llvm.load %ptr : !llvm.ptr -> f32
  %sin = math.sin %ld : f32
  return %sin : f32
}

func.func @dexec_region_ptr(%x: !llvm.ptr, %dx: !llvm.ptr, %dres: f32) {
  enzyme.autodiff @exec_region_ptr(%x, %dx, %dres) <{
    activity = [#enzyme.activity<enzyme_dup>],
    ret_activity = [#enzyme.activity<enzyme_activenoneed>]
  }> : (!llvm.ptr, !llvm.ptr, f32) -> ()
  return
}

// CHECK-LABEL: func.func private @diffeexec_region_ptr(
// CHECK-SAME: %[[X:.+]]: !llvm.ptr, %[[DX:.+]]: !llvm.ptr, %[[DRES:.+]]: f32) {
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[FWD:.+]]:3 = scf.execute_region -> (!llvm.ptr, !llvm.ptr, f32) no_inline {
// CHECK-NEXT:      %[[DGEP:.+]] = llvm.getelementptr %[[DX]][4] : (!llvm.ptr) -> !llvm.ptr, f32
// CHECK-NEXT:      %[[GEP:.+]] = llvm.getelementptr %[[X]][4] : (!llvm.ptr) -> !llvm.ptr, f32
// CHECK-NEXT:      %[[VAL:.+]] = llvm.load %[[X]] : !llvm.ptr -> f32
// CHECK-NEXT:      %[[COS:.+]] = math.cos %[[VAL]] : f32
// CHECK-NEXT:      llvm.store %[[COS]], %[[X]] : f32, !llvm.ptr
// CHECK-NEXT:      scf.yield %[[GEP]], %[[DGEP]], %[[VAL]] : !llvm.ptr, !llvm.ptr, f32
// CHECK:         %[[LD:.+]] = llvm.load %[[FWD]]#0 : !llvm.ptr -> f32
// CHECK-NEXT:    %[[COS1:.+]] = math.cos %[[LD]] fastmath<fast> : f32
// CHECK-NEXT:    %[[M0:.+]] = arith.mulf %[[DRES]], %[[COS1]] fastmath<fast> : f32
// CHECK-NEXT:    %[[DL:.+]] = llvm.load %[[FWD]]#1 : !llvm.ptr -> f32
// CHECK-NEXT:    %[[ADD0:.+]] = arith.addf %[[DL]], %[[M0]] fastmath<fast> : f32
// CHECK-NEXT:    llvm.store %[[ADD0]], %[[FWD]]#1 : f32, !llvm.ptr
// CHECK-NEXT:    scf.execute_region no_inline {
// CHECK-NEXT:      %[[LDX:.+]] = llvm.load %[[DX]] : !llvm.ptr -> f32
// CHECK-NEXT:      llvm.store %[[ZERO]], %[[DX]] : f32, !llvm.ptr
// CHECK-NEXT:      %[[SIN:.+]] = math.sin %[[FWD]]#2 fastmath<fast> : f32
// CHECK-NEXT:      %[[NEG:.+]] = arith.negf %[[SIN]] fastmath<fast> : f32
// CHECK-NEXT:      %[[M1:.+]] = arith.mulf %[[LDX]], %[[NEG]] fastmath<fast> : f32
// CHECK-NEXT:      %[[LDX2:.+]] = llvm.load %[[DX]] : !llvm.ptr -> f32
// CHECK-NEXT:      %[[ADD1:.+]] = arith.addf %[[LDX2]], %[[M1]] fastmath<fast> : f32
// CHECK-NEXT:      llvm.store %[[ADD1]], %[[DX]] : f32, !llvm.ptr
// CHECK-NEXT:      scf.yield
// CHECK:         return
