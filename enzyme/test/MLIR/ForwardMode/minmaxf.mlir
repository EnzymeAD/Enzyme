// RUN: %eopt --split-input-file --enzyme %s | FileCheck %s

func.func @maximumf(%a: f64, %b: f64) -> f64 {
  %res = arith.maximumf %a, %b : f64
  return %res : f64
}

func.func @dmaximumf(%a: f64, %da: f64, %b: f64, %db: f64) -> f64 {
  %r = enzyme.fwddiff @maximumf(%a, %da, %b, %db)
    {
      activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
      ret_activity=[#enzyme<activity enzyme_dupnoneed>]
    } : (f64, f64, f64, f64) -> (f64)
  return %r : f64
}

// CHECK:  func.func private @fwddiffemaximumf(%[[a:.+]]: f64, %[[da:.+]]: f64, %[[b:.+]]: f64, %[[db:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[lt:.+]] = arith.cmpf olt, %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[one:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[sa:.+]] = math.copysign %[[one]], %[[a]] fastmath<fast> : f64
// CHECK-NEXT:    %[[mone:.+]] = arith.constant -1.000000e+00 : f64
// CHECK-NEXT:    %[[aneg0:.+]] = arith.cmpf oeq, %[[sa]], %[[mone]] fastmath<fast> : f64
// CHECK-NEXT:    %[[one2:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[sb:.+]] = math.copysign %[[one2]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[one3:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[bpos0:.+]] = arith.cmpf oeq, %[[sb]], %[[one3]] fastmath<fast> : f64
// CHECK-NEXT:    %[[szero:.+]] = arith.andi %[[aneg0]], %[[bpos0]] : i1
// CHECK-NEXT:    %[[zero2:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[ta0:.+]] = arith.select %[[szero]], %[[zero2]], %[[da]] : f64
// CHECK-NEXT:    %[[ta:.+]] = arith.select %[[lt]], %[[zero]], %[[ta0]] : f64
// CHECK-NEXT:    %[[lt2:.+]] = arith.cmpf olt, %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[one4:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[sa2:.+]] = math.copysign %[[one4]], %[[a]] fastmath<fast> : f64
// CHECK-NEXT:    %[[mone2:.+]] = arith.constant -1.000000e+00 : f64
// CHECK-NEXT:    %[[aneg02:.+]] = arith.cmpf oeq, %[[sa2]], %[[mone2]] fastmath<fast> : f64
// CHECK-NEXT:    %[[one5:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[sb2:.+]] = math.copysign %[[one5]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[one6:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[bpos02:.+]] = arith.cmpf oeq, %[[sb2]], %[[one6]] fastmath<fast> : f64
// CHECK-NEXT:    %[[szero2:.+]] = arith.andi %[[aneg02]], %[[bpos02]] : i1
// CHECK-NEXT:    %[[zero3:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[tb0:.+]] = arith.select %[[szero2]], %[[db]], %[[zero3]] : f64
// CHECK-NEXT:    %[[tb:.+]] = arith.select %[[lt2]], %[[db]], %[[tb0]] : f64
// CHECK-NEXT:    %[[res:.+]] = arith.addf %[[ta]], %[[tb]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = arith.maximumf %[[a]], %[[b]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @minimumf(%a: f64, %b: f64) -> f64 {
  %res = arith.minimumf %a, %b : f64
  return %res : f64
}

func.func @dminimumf(%a: f64, %da: f64, %b: f64, %db: f64) -> f64 {
  %r = enzyme.fwddiff @minimumf(%a, %da, %b, %db)
    {
      activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
      ret_activity=[#enzyme<activity enzyme_dupnoneed>]
    } : (f64, f64, f64, f64) -> (f64)
  return %r : f64
}

// CHECK:  func.func private @fwddiffeminimumf(%[[a:.+]]: f64, %[[da:.+]]: f64, %[[b:.+]]: f64, %[[db:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[lt:.+]] = arith.cmpf olt, %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[one:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[sa:.+]] = math.copysign %[[one]], %[[a]] fastmath<fast> : f64
// CHECK-NEXT:    %[[mone:.+]] = arith.constant -1.000000e+00 : f64
// CHECK-NEXT:    %[[aneg0:.+]] = arith.cmpf oeq, %[[sa]], %[[mone]] fastmath<fast> : f64
// CHECK-NEXT:    %[[one2:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[sb:.+]] = math.copysign %[[one2]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[one3:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[bpos0:.+]] = arith.cmpf oeq, %[[sb]], %[[one3]] fastmath<fast> : f64
// CHECK-NEXT:    %[[szero:.+]] = arith.andi %[[aneg0]], %[[bpos0]] : i1
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[ta0:.+]] = arith.select %[[szero]], %[[da]], %[[zero]] : f64
// CHECK-NEXT:    %[[ta:.+]] = arith.select %[[lt]], %[[da]], %[[ta0]] : f64
// CHECK-NEXT:    %[[lt2:.+]] = arith.cmpf olt, %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[zero2:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[one4:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[sa2:.+]] = math.copysign %[[one4]], %[[a]] fastmath<fast> : f64
// CHECK-NEXT:    %[[mone2:.+]] = arith.constant -1.000000e+00 : f64
// CHECK-NEXT:    %[[aneg02:.+]] = arith.cmpf oeq, %[[sa2]], %[[mone2]] fastmath<fast> : f64
// CHECK-NEXT:    %[[one5:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[sb2:.+]] = math.copysign %[[one5]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[one6:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[bpos02:.+]] = arith.cmpf oeq, %[[sb2]], %[[one6]] fastmath<fast> : f64
// CHECK-NEXT:    %[[szero2:.+]] = arith.andi %[[aneg02]], %[[bpos02]] : i1
// CHECK-NEXT:    %[[zero3:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[tb0:.+]] = arith.select %[[szero2]], %[[zero3]], %[[db]] : f64
// CHECK-NEXT:    %[[tb:.+]] = arith.select %[[lt2]], %[[zero2]], %[[tb0]] : f64
// CHECK-NEXT:    %[[res:.+]] = arith.addf %[[ta]], %[[tb]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = arith.minimumf %[[a]], %[[b]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }
