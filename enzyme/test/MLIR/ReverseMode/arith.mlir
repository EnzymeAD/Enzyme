// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s

func.func @select(%c: i1, %a: f64, %b: f64) -> f64 {
  %res = arith.select %c, %a, %b : f64
  return %res : f64
}

func.func @dselect(%c: i1, %a: f64, %b: f64, %dr: f64) -> (f64, f64) {
  %0:2 = enzyme.autodiff @select(%c, %a, %b, %dr)
    {
      activity=[#enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (i1, f64, f64, f64) -> (f64, f64)
  return %0#0, %0#1 : f64, f64
}

// CHECK: func.func private @diffeselect(%[[c:.+]]: i1, %[[a:.+]]: f64, %[[b:.+]]: f64, %[[dr:.+]]: f64) -> (f64, f64) {
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[da:.+]] = arith.select %[[c]], %[[dr]], %[[zero]] : f64
// CHECK-NEXT:    %[[db:.+]] = arith.select %[[c]], %[[zero]], %[[dr]] : f64
// CHECK-NEXT:    return %[[da]], %[[db]] : f64, f64
// CHECK-NEXT:  }

// -----

func.func @maxnumf(%a: f64, %b: f64) -> f64 {
  %res = arith.maxnumf %a, %b : f64
  return %res : f64
}

func.func @dmaxnumf(%a: f64, %b: f64, %dr: f64) -> (f64, f64) {
  %0:2 = enzyme.autodiff @maxnumf(%a, %b, %dr)
    {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64, f64) -> (f64, f64)
  return %0#0, %0#1 : f64, f64
}

// CHECK: func.func private @diffemaxnumf(%[[a:.+]]: f64, %[[b:.+]]: f64, %[[dr:.+]]: f64) -> (f64, f64) {
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[cmp1:.+]] = arith.cmpf olt, %[[a]], %[[b]] : f64
// CHECK-NEXT:    %[[da:.+]] = arith.select %[[cmp1]], %[[zero]], %[[dr]] : f64
// CHECK-NEXT:    %[[cmp2:.+]] = arith.cmpf olt, %[[a]], %[[b]] : f64
// CHECK-NEXT:    %[[db:.+]] = arith.select %[[cmp2]], %[[dr]], %[[zero]] : f64
// CHECK-NEXT:    return %[[da]], %[[db]] : f64, f64
// CHECK-NEXT:  }

// -----

func.func @select_ptr(%c: i1, %a: memref<f64>, %b: memref<f64>) -> f64 {
  %ptr = arith.select %c, %a, %b : memref<f64>
  %val = memref.load %ptr[] : memref<f64>
  return %val : f64
}

func.func @dselect_ptr(%c: i1, %a: memref<f64>, %da: memref<f64>, %b: memref<f64>, %db: memref<f64>, %dr: f64) {
  enzyme.autodiff @select_ptr(%c, %a, %da, %b, %db, %dr)
    {
      activity=[#enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (i1, memref<f64>, memref<f64>, memref<f64>, memref<f64>, f64) -> ()
  return
}

// CHECK: func.func private @diffeselect_ptr(%[[c:.+]]: i1, %[[a:.+]]: memref<f64>, %[[da:.+]]: memref<f64>, %[[b:.+]]: memref<f64>, %[[db:.+]]: memref<f64>, %[[dr:.+]]: f64) {
// CHECK-NEXT:    %[[dptr:.+]] = arith.select %[[c]], %[[da]], %[[db]] : memref<f64>
// CHECK-NEXT:    %[[v0:.+]] = memref.load %[[dptr]][] : memref<f64>
// CHECK-NEXT:    %[[v1:.+]] = arith.addf %[[v0]], %[[dr]] : f64
// CHECK-NEXT:    memref.store %[[v1]], %[[dptr]][] : memref<f64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

