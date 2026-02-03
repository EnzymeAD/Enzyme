// RUN: %eopt %s --split-input-file --pass-pipeline="builtin.module(enzyme,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math)" | FileCheck %s

func.func @nested_if(%arg0: f64, %arg1: i1, %arg2: i1) -> f64 {
  %cst = arith.constant 4.000000e+00 : f64
  %0 = scf.if %arg1 -> (f64) {
    %1 = arith.mulf %arg0, %arg0 : f64
    scf.yield %1 : f64
  } else {
    %1 = scf.if %arg2 -> (f64) {
      %3 = math.cos %arg0 : f64
      scf.yield %3 : f64
    } else {
      scf.yield %cst : f64
    }
    %2 = arith.addf %1, %arg0 : f64
    scf.yield %2 : f64
  }
  return %0 : f64
}

func.func @dnested_if(%arg0: f64, %arg1: i1, %arg2: i1, %dres: f64) -> f64 {
  %res = enzyme.autodiff @nested_if(%arg0, %arg1, %arg2, %dres)
    {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (f64, i1, i1, f64) -> f64
  return %res : f64
}
// CHECK: func.func private @diffenested_if(%arg0: f64, %arg1: i1, %arg2: i1, %arg3: f64) -> f64 {
// CHECK-NEXT:  %0 = scf.if %arg1 -> (f64) {
// CHECK-NEXT:      %1 = arith.mulf %arg3, %arg0 : f64
// CHECK-NEXT:      %2 = arith.mulf %arg3, %arg0 : f64
// CHECK-NEXT:      %3 = arith.addf %1, %2 : f64
// CHECK-NEXT:      scf.yield %3 : f64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %1 = scf.if %arg2 -> (f64) {
// CHECK-NEXT:        %2 = math.sin %arg0 : f64
// CHECK-NEXT:        %3 = arith.negf %2 : f64
// CHECK-NEXT:        %4 = arith.mulf %arg3, %3 : f64
// CHECK-NEXT:        %5 = arith.addf %arg3, %4 : f64
// CHECK-NEXT:        scf.yield %5 : f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %arg3 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %1 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0 : f64
// CHECK-NEXT:  }

// -----

func.func @some_res_inactive(%arg0: f64, %arg1: i1) -> (f64, i64) {
  %0:2 = scf.if %arg1 -> (f64, i64) {
    %1 = arith.mulf %arg0, %arg0 : f64
    %2 = arith.fptosi %1 : f64 to i64
    scf.yield %1, %2 : f64, i64
  } else {
    %1 = math.cos %arg0 : f64
    %2 = arith.addf %1, %arg0 : f64
    %3 = arith.fptosi %2 : f64 to i64
    scf.yield %2, %3 : f64, i64
  }
  return %0#0, %0#1 : f64, i64
}

func.func @dsome_res_inactive(%x: f64, %cond: i1, %dres: f64) -> f64 {
  %true = arith.constant true
  %res = enzyme.autodiff @some_res_inactive(%x, %cond, %dres)
    {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_constnoneed>]
    } : (f64, i1, f64) -> (f64)
  return %res : f64
}
// CHECK: func.func private @diffesome_res_inactive(%arg0: f64, %arg1: i1, %arg2: f64) -> f64 {
// CHECK-NEXT:  %0 = scf.if %arg1 -> (f64) {
// CHECK-NEXT:      %1 = arith.mulf %arg2, %arg0 : f64
// CHECK-NEXT:      %2 = arith.mulf %arg2, %arg0 : f64
// CHECK-NEXT:      %3 = arith.addf %1, %2 : f64
// CHECK-NEXT:      scf.yield %3 : f64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %1 = math.sin %arg0 : f64
// CHECK-NEXT:      %2 = arith.negf %1 : f64
// CHECK-NEXT:      %3 = arith.mulf %arg2, %2 : f64
// CHECK-NEXT:      %4 = arith.addf %arg2, %3 : f64
// CHECK-NEXT:      scf.yield %4 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0 : f64
// CHECK-NEXT:  }

// -----

func.func private @if_overwrite(%cond: i1, %x: memref<f32>) {
  scf.if %cond {
    %val = memref.load %x[] : memref<f32>
    %cos = math.cos %val : f32
    memref.store %cos, %x[] : memref<f32>
  }
  return
}

func.func @dif_overwrite(%cond: i1, %x: memref<f32>, %dx: memref<f32>) {
  enzyme.autodiff @if_overwrite(%cond, %x, %dx) {
    activity=[#enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>],
    ret_activity=[]
  } : (i1, memref<f32>, memref<f32>) -> ()
  return
}

// CHECK-LABEL:   func.func private @diffeif_overwrite(
// CHECK-SAME:      %[[ARG0:.*]]: i1,
// CHECK-SAME:      %[[ARG1:.*]]: memref<f32>,
// CHECK-SAME:      %[[ARG2:.*]]: memref<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[IF_0:.*]] = scf.if %[[ARG0]] -> (f32) {
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[ARG1]][] : memref<f32>
// CHECK:             %[[COS_0:.*]] = math.cos %[[LOAD_0]] : f32
// CHECK:             memref.store %[[COS_0]], %[[ARG1]][] : memref<f32>
// CHECK:             scf.yield %[[LOAD_0]] : f32
// CHECK:           } else {
// CHECK:             scf.yield %[[CONSTANT_0]] : f32
// CHECK:           }
// CHECK:           scf.if %[[ARG0]] {
// CHECK:             %[[LOAD_1:.*]] = memref.load %[[ARG2]][] : memref<f32>
// CHECK:             memref.store %[[CONSTANT_0]], %[[ARG2]][] : memref<f32>
// CHECK:             %[[SIN_0:.*]] = math.sin %[[IF_0]] : f32
// CHECK:             %[[NEGF_0:.*]] = arith.negf %[[SIN_0]] : f32
// CHECK:             %[[MULF_0:.*]] = arith.mulf %[[LOAD_1]], %[[NEGF_0]] : f32
// CHECK:             %[[LOAD_2:.*]] = memref.load %[[ARG2]][] : memref<f32>
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[LOAD_2]], %[[MULF_0]] : f32
// CHECK:             memref.store %[[ADDF_0]], %[[ARG2]][] : memref<f32>
// CHECK:           } else {
// CHECK:           }
// CHECK:           return
// CHECK:         }
