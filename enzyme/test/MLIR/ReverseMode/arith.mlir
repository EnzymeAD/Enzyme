// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s

func.func @select(%c: i1, %a: f64, %b: f64) -> f64 {
  %res = arith.select %c, %a, %b : f64
  return %res : f64
}

func.func @dselect(%c: i1, %a: f64, %b: f64, %dr: f64) -> (f64, f64) {
  %0:2 = enzyme.autodiff @select(%c, %a, %b, %dr)
    <{
      activity=[#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_activenoneed>]
    }> : (i1, f64, f64, f64) -> (f64, f64)
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
    <{
      activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_activenoneed>]
    }> : (f64, f64, f64) -> (f64, f64)
  return %0#0, %0#1 : f64, f64
}

// CHECK: func.func private @diffemaxnumf(%[[a:.+]]: f64, %[[b:.+]]: f64, %[[dr:.+]]: f64) -> (f64, f64) {
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[cmp1:.+]] = arith.cmpf olt, %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[da:.+]] = arith.select %[[cmp1]], %[[zero]], %[[dr]] : f64
// CHECK-NEXT:    %[[cmp2:.+]] = arith.cmpf olt, %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[db:.+]] = arith.select %[[cmp2]], %[[dr]], %[[zero]] : f64
// CHECK-NEXT:    return %[[da]], %[[db]] : f64, f64
// CHECK-NEXT:  }

// -----

func.func @minnumf(%a: f64, %b: f64) -> f64 {
  %res = arith.minnumf %a, %b : f64
  return %res : f64
}

func.func @dminnumf(%a: f64, %b: f64, %dr: f64) -> (f64, f64) {
  %0:2 = enzyme.autodiff @minnumf(%a, %b, %dr)
    <{
      activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_activenoneed>]
    }> : (f64, f64, f64) -> (f64, f64)
  return %0#0, %0#1 : f64, f64
}

// CHECK: func.func private @diffeminnumf(%[[a:.+]]: f64, %[[b:.+]]: f64, %[[dr:.+]]: f64) -> (f64, f64) {
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[cmp1:.+]] = arith.cmpf ogt, %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[da:.+]] = arith.select %[[cmp1]], %[[zero]], %[[dr]] : f64
// CHECK-NEXT:    %[[cmp2:.+]] = arith.cmpf ogt, %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[db:.+]] = arith.select %[[cmp2]], %[[dr]], %[[zero]] : f64
// CHECK-NEXT:    return %[[da]], %[[db]] : f64, f64
// CHECK-NEXT:  }

// -----

func.func @minimumf(%a: f64, %b: f64) -> f64 {
  %res = arith.minimumf %a, %b : f64
  return %res : f64
}

func.func @dminimumf(%a: f64, %b: f64, %dr: f64) -> (f64, f64) {
  %0:2 = enzyme.autodiff @minimumf(%a, %b, %dr)
    <{
      activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_activenoneed>]
    }> : (f64, f64, f64) -> (f64, f64)
  return %0#0, %0#1 : f64, f64
}

// CHECK: func.func private @diffeminimumf(%[[a:.+]]: f64, %[[b:.+]]: f64, %[[dr:.+]]: f64) -> (f64, f64) {
// CHECK-DAG:    %[[mone:.+]] = arith.constant -1.000000e+00 : f64
// CHECK-DAG:    %[[one:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK:    %[[lt:.+]] = arith.cmpf olt, %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[sa:.+]] = math.copysign %[[one]], %[[a]] fastmath<fast> : f64
// CHECK-NEXT:    %[[aneg0:.+]] = arith.cmpf oeq, %[[sa]], %[[mone]] fastmath<fast> : f64
// CHECK-NEXT:    %[[sb:.+]] = math.copysign %[[one]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[bpos0:.+]] = arith.cmpf oeq, %[[sb]], %[[one]] fastmath<fast> : f64
// CHECK-NEXT:    %[[szero:.+]] = arith.andi %[[aneg0]], %[[bpos0]] : i1
// CHECK-NEXT:    %[[da0:.+]] = arith.select %[[szero]], %[[dr]], %[[zero]] : f64
// CHECK-NEXT:    %[[da:.+]] = arith.select %[[lt]], %[[dr]], %[[da0]] : f64
// CHECK-NEXT:    %[[lt2:.+]] = arith.cmpf olt, %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[sa2:.+]] = math.copysign %[[one]], %[[a]] fastmath<fast> : f64
// CHECK-NEXT:    %[[aneg02:.+]] = arith.cmpf oeq, %[[sa2]], %[[mone]] fastmath<fast> : f64
// CHECK-NEXT:    %[[sb2:.+]] = math.copysign %[[one]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[bpos02:.+]] = arith.cmpf oeq, %[[sb2]], %[[one]] fastmath<fast> : f64
// CHECK-NEXT:    %[[szero2:.+]] = arith.andi %[[aneg02]], %[[bpos02]] : i1
// CHECK-NEXT:    %[[db0:.+]] = arith.select %[[szero2]], %[[zero]], %[[dr]] : f64
// CHECK-NEXT:    %[[db:.+]] = arith.select %[[lt2]], %[[zero]], %[[db0]] : f64
// CHECK-NEXT:    return %[[da]], %[[db]] : f64, f64
// CHECK-NEXT:  }

// -----

func.func @maximumf(%a: f64, %b: f64) -> f64 {
  %res = arith.maximumf %a, %b : f64
  return %res : f64
}

func.func @dmaximumf(%a: f64, %b: f64, %dr: f64) -> (f64, f64) {
  %0:2 = enzyme.autodiff @maximumf(%a, %b, %dr)
    <{
      activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_activenoneed>]
    }> : (f64, f64, f64) -> (f64, f64)
  return %0#0, %0#1 : f64, f64
}

// CHECK: func.func private @diffemaximumf(%[[a:.+]]: f64, %[[b:.+]]: f64, %[[dr:.+]]: f64) -> (f64, f64) {
// CHECK-DAG:    %[[mone:.+]] = arith.constant -1.000000e+00 : f64
// CHECK-DAG:    %[[one:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK:    %[[lt:.+]] = arith.cmpf olt, %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[sa:.+]] = math.copysign %[[one]], %[[a]] fastmath<fast> : f64
// CHECK-NEXT:    %[[aneg0:.+]] = arith.cmpf oeq, %[[sa]], %[[mone]] fastmath<fast> : f64
// CHECK-NEXT:    %[[sb:.+]] = math.copysign %[[one]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[bpos0:.+]] = arith.cmpf oeq, %[[sb]], %[[one]] fastmath<fast> : f64
// CHECK-NEXT:    %[[szero:.+]] = arith.andi %[[aneg0]], %[[bpos0]] : i1
// CHECK-NEXT:    %[[da0:.+]] = arith.select %[[szero]], %[[zero]], %[[dr]] : f64
// CHECK-NEXT:    %[[da:.+]] = arith.select %[[lt]], %[[zero]], %[[da0]] : f64
// CHECK-NEXT:    %[[lt2:.+]] = arith.cmpf olt, %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[sa2:.+]] = math.copysign %[[one]], %[[a]] fastmath<fast> : f64
// CHECK-NEXT:    %[[aneg02:.+]] = arith.cmpf oeq, %[[sa2]], %[[mone]] fastmath<fast> : f64
// CHECK-NEXT:    %[[sb2:.+]] = math.copysign %[[one]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[bpos02:.+]] = arith.cmpf oeq, %[[sb2]], %[[one]] fastmath<fast> : f64
// CHECK-NEXT:    %[[szero2:.+]] = arith.andi %[[aneg02]], %[[bpos02]] : i1
// CHECK-NEXT:    %[[db0:.+]] = arith.select %[[szero2]], %[[dr]], %[[zero]] : f64
// CHECK-NEXT:    %[[db:.+]] = arith.select %[[lt2]], %[[dr]], %[[db0]] : f64
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
    <{
      activity=[#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity=[#enzyme.activity<enzyme_activenoneed>]
    }> : (i1, memref<f64>, memref<f64>, memref<f64>, memref<f64>, f64) -> ()
  return
}

// CHECK: func.func private @diffeselect_ptr(%[[c:.+]]: i1, %[[a:.+]]: memref<f64>, %[[da:.+]]: memref<f64>, %[[b:.+]]: memref<f64>, %[[db:.+]]: memref<f64>, %[[dr:.+]]: f64) {
// CHECK-NEXT:    %[[dptr:.+]] = arith.select %[[c]], %[[da]], %[[db]] : memref<f64>
// CHECK-NEXT:    %[[v0:.+]] = memref.load %[[dptr]][] : memref<f64>
// CHECK-NEXT:    %[[v1:.+]] = arith.addf %[[v0]], %[[dr]] fastmath<fast> : f64
// CHECK-NEXT:    memref.store %[[v1]], %[[dptr]][] : memref<f64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----

func.func @remf(%x: f64, %y: f64) -> f64 {
  %res = arith.remf %x, %y : f64
  return %res : f64
}

func.func @dremf(%x: f64, %y: f64, %dr: f64) -> (f64, f64) {
  %0:2 = enzyme.autodiff @remf(%x, %y, %dr)
    <{
      activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_activenoneed>]
    }> : (f64, f64, f64) -> (f64, f64)
  return %0#0, %0#1 : f64, f64
}

// CHECK: func.func private @differemf(%[[x:.+]]: f64, %[[y:.+]]: f64, %[[dr:.+]]: f64) -> (f64, f64) {
// CHECK-NEXT:    %[[div:.+]] = arith.divf %[[x]], %[[y]] fastmath<fast> : f64
// CHECK-NEXT:    %[[abs:.+]] = math.absf %[[div]] fastmath<fast> : f64
// CHECK-NEXT:    %[[floor:.+]] = math.floor %[[abs]] fastmath<fast> : f64
// CHECK-NEXT:    %[[cs:.+]] = math.copysign %[[floor]], %[[div]] fastmath<fast> : f64
// CHECK-NEXT:    %[[neg:.+]] = arith.negf %[[cs]] fastmath<fast> : f64
// CHECK-NEXT:    %[[dy:.+]] = arith.mulf %[[dr]], %[[neg]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[dr]], %[[dy]] : f64, f64
// CHECK-NEXT:  }

// -----

// Batched reverse mode of arith.maxnumf / arith.maximumf / arith.minimumf.

func.func @maximumf(%a: f64, %b: f64) -> f64 {
  %res = arith.maximumf %a, %b : f64
  return %res : f64
}

func.func @dmaximumf(%a: f64, %b: f64, %dr: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
  %0:2 = enzyme.autodiff @maximumf(%a, %b, %dr)
    <{
      activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_activenoneed>],
      width=2
    }> : (f64, f64, tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>)
  return %0#0, %0#1 : tensor<2xf64>, tensor<2xf64>
}

// CHECK:  func.func private @diffe2maximumf(%arg0: f64, %arg1: f64, %arg2: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
// CHECK-NEXT:    %cst = arith.constant -1.000000e+00 : f64
// CHECK-NEXT:    %cst_0 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %cst_1 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.cmpf olt, %arg0, %arg1 fastmath<fast> : f64
// CHECK-NEXT:    %1 = math.copysign %cst_0, %arg0 fastmath<fast> : f64
// CHECK-NEXT:    %2 = arith.cmpf oeq, %1, %cst fastmath<fast> : f64
// CHECK-NEXT:    %3 = math.copysign %cst_0, %arg1 fastmath<fast> : f64
// CHECK-NEXT:    %4 = arith.cmpf oeq, %3, %cst_0 fastmath<fast> : f64
// CHECK-NEXT:    %5 = arith.andi %2, %4 : i1
// CHECK-NEXT:    %6 = "enzyme.broadcast"(%cst_1) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %7 = "enzyme.broadcast"(%5) <{shape = array<i64: 2>}> : (i1) -> tensor<2xi1>
// CHECK-NEXT:    %8 = arith.select %7, %6, %arg2 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:    %9 = "enzyme.broadcast"(%cst_1) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %10 = "enzyme.broadcast"(%0) <{shape = array<i64: 2>}> : (i1) -> tensor<2xi1>
// CHECK-NEXT:    %11 = arith.select %10, %9, %8 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:    %12 = arith.cmpf olt, %arg0, %arg1 fastmath<fast> : f64
// CHECK-NEXT:    %13 = math.copysign %cst_0, %arg0 fastmath<fast> : f64
// CHECK-NEXT:    %14 = arith.cmpf oeq, %13, %cst fastmath<fast> : f64
// CHECK-NEXT:    %15 = math.copysign %cst_0, %arg1 fastmath<fast> : f64
// CHECK-NEXT:    %16 = arith.cmpf oeq, %15, %cst_0 fastmath<fast> : f64
// CHECK-NEXT:    %17 = arith.andi %14, %16 : i1
// CHECK-NEXT:    %18 = "enzyme.broadcast"(%cst_1) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %19 = "enzyme.broadcast"(%17) <{shape = array<i64: 2>}> : (i1) -> tensor<2xi1>
// CHECK-NEXT:    %20 = arith.select %19, %arg2, %18 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:    %21 = "enzyme.broadcast"(%12) <{shape = array<i64: 2>}> : (i1) -> tensor<2xi1>
// CHECK-NEXT:    %22 = arith.select %21, %arg2, %20 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:    return %11, %22 : tensor<2xf64>, tensor<2xf64>
// CHECK-NEXT:  }

// -----

func.func @minimumf(%a: f64, %b: f64) -> f64 {
  %res = arith.minimumf %a, %b : f64
  return %res : f64
}

func.func @dminimumf(%a: f64, %b: f64, %dr: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
  %0:2 = enzyme.autodiff @minimumf(%a, %b, %dr)
    <{
      activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_activenoneed>],
      width=2
    }> : (f64, f64, tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>)
  return %0#0, %0#1 : tensor<2xf64>, tensor<2xf64>
}

// CHECK:  func.func private @diffe2minimumf(%arg0: f64, %arg1: f64, %arg2: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %cst_0 = arith.constant -1.000000e+00 : f64
// CHECK-NEXT:    %cst_1 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.cmpf olt, %arg0, %arg1 fastmath<fast> : f64
// CHECK-NEXT:    %1 = math.copysign %cst_1, %arg0 fastmath<fast> : f64
// CHECK-NEXT:    %2 = arith.cmpf oeq, %1, %cst_0 fastmath<fast> : f64
// CHECK-NEXT:    %3 = math.copysign %cst_1, %arg1 fastmath<fast> : f64
// CHECK-NEXT:    %4 = arith.cmpf oeq, %3, %cst_1 fastmath<fast> : f64
// CHECK-NEXT:    %5 = arith.andi %2, %4 : i1
// CHECK-NEXT:    %6 = "enzyme.broadcast"(%cst) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %7 = "enzyme.broadcast"(%5) <{shape = array<i64: 2>}> : (i1) -> tensor<2xi1>
// CHECK-NEXT:    %8 = arith.select %7, %arg2, %6 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:    %9 = "enzyme.broadcast"(%0) <{shape = array<i64: 2>}> : (i1) -> tensor<2xi1>
// CHECK-NEXT:    %10 = arith.select %9, %arg2, %8 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:    %11 = arith.cmpf olt, %arg0, %arg1 fastmath<fast> : f64
// CHECK-NEXT:    %12 = math.copysign %cst_1, %arg0 fastmath<fast> : f64
// CHECK-NEXT:    %13 = arith.cmpf oeq, %12, %cst_0 fastmath<fast> : f64
// CHECK-NEXT:    %14 = math.copysign %cst_1, %arg1 fastmath<fast> : f64
// CHECK-NEXT:    %15 = arith.cmpf oeq, %14, %cst_1 fastmath<fast> : f64
// CHECK-NEXT:    %16 = arith.andi %13, %15 : i1
// CHECK-NEXT:    %17 = "enzyme.broadcast"(%cst) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %18 = "enzyme.broadcast"(%16) <{shape = array<i64: 2>}> : (i1) -> tensor<2xi1>
// CHECK-NEXT:    %19 = arith.select %18, %17, %arg2 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:    %20 = "enzyme.broadcast"(%cst) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %21 = "enzyme.broadcast"(%11) <{shape = array<i64: 2>}> : (i1) -> tensor<2xi1>
// CHECK-NEXT:    %22 = arith.select %21, %20, %19 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:    return %10, %22 : tensor<2xf64>, tensor<2xf64>
// CHECK-NEXT:  }

// -----

func.func @maxnumf(%a: f64, %b: f64) -> f64 {
  %res = arith.maxnumf %a, %b : f64
  return %res : f64
}

func.func @dmaxnumf(%a: f64, %b: f64, %dr: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
  %0:2 = enzyme.autodiff @maxnumf(%a, %b, %dr)
    <{
      activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_activenoneed>],
      width=2
    }> : (f64, f64, tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>)
  return %0#0, %0#1 : tensor<2xf64>, tensor<2xf64>
}

// CHECK:  func.func private @diffe2maxnumf(%arg0: f64, %arg1: f64, %arg2: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.cmpf olt, %arg0, %arg1 fastmath<fast> : f64
// CHECK-NEXT:    %1 = "enzyme.broadcast"(%cst) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %2 = "enzyme.broadcast"(%0) <{shape = array<i64: 2>}> : (i1) -> tensor<2xi1>
// CHECK-NEXT:    %3 = arith.select %2, %1, %arg2 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:    %4 = arith.cmpf olt, %arg0, %arg1 fastmath<fast> : f64
// CHECK-NEXT:    %5 = "enzyme.broadcast"(%cst) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %6 = "enzyme.broadcast"(%4) <{shape = array<i64: 2>}> : (i1) -> tensor<2xi1>
// CHECK-NEXT:    %7 = arith.select %6, %arg2, %5 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:    return %3, %7 : tensor<2xf64>, tensor<2xf64>
// CHECK-NEXT:  }
