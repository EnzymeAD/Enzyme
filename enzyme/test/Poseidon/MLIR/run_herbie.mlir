// REQUIRES: poseidon-mlir
// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_mlir_opt --mlir-disable-threading \
// RUN:   --poseidon-rewrite="run-herbie herbie-binary=%S/Inputs/run_herbie/fake-herbie.sh herbie-threads=1 herbie-cache=%t/c" \
// RUN:   %s -o %t/first.mlir 2> %t/first.err
// RUN: FileCheck %s < %t/first.mlir
// RUN: FileCheck %s --check-prefix=FIRST < %t/first.err
// RUN: %poseidon_mlir_opt --mlir-disable-threading \
// RUN:   --poseidon-rewrite="run-herbie herbie-binary=%S/Inputs/run_herbie/fake-herbie.sh herbie-threads=1 herbie-cache=%t/c" \
// RUN:   %s -o %t/again.mlir 2> %t/again.err
// RUN: diff %t/first.mlir %t/again.mlir
// RUN: FileCheck %s --check-prefix=AGAIN < %t/again.err
// RUN: %poseidon_mlir_opt \
// RUN:   --poseidon-rewrite="run-herbie herbie-binary=%S/Inputs/run_herbie/fake-herbie.sh herbie-threads=2 herbie-cache=%t/c2" \
// RUN:   %s 2> /dev/null | FileCheck %s
// RUN: not %poseidon_mlir_opt \
// RUN:   --poseidon-rewrite="run-herbie herbie-binary=%t/no-such-herbie herbie-cache=%t/c3" \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: %poseidon_mlir_opt \
// RUN:   --poseidon-rewrite="run-herbie herbie-binary=%t/no-such-herbie herbie-results=%S/Inputs/run_herbie/results.json" \
// RUN:   %s | FileCheck %s

// With run-herbie, Herbie is run on each function's chains, one core per
// chain output, and its results are cached under the cores and how Herbie
// was run. Here a script stands in for Herbie and gives fixed answers.

// The same rewrites come back however many Herbie processes the cores are
// split across, and from the cache.
// CHECK-LABEL: func.func @log_one_plus
// CHECK:         %[[R:.*]] = math.log1p %arg0 : tensor<8xf64>
// CHECK-NEXT:    return %[[R]]
func.func @log_one_plus(%x: tensor<8xf64>) -> tensor<8xf64> {
  %one = arith.constant dense<1.0> : tensor<8xf64>
  %0 = arith.addf %one, %x : tensor<8xf64>
  %1 = math.log %0 : tensor<8xf64>
  return %1 : tensor<8xf64>
}

// Two outputs: both cores go to Herbie in one run.
// CHECK-LABEL: func.func @two_outputs
// CHECK:         %[[E:.*]] = math.exp %arg0
// CHECK-NEXT:    %[[M:.*]] = math.expm1 %arg0
// CHECK-NEXT:    %[[S:.*]] = math.sqrt %[[M]]
// CHECK-NEXT:    return %[[E]], %[[S]]
func.func @two_outputs(%x: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
  %one = arith.constant dense<1.0> : tensor<4xf64>
  %0 = math.exp %x : tensor<4xf64>
  %1 = arith.subf %0, %one : tensor<4xf64>
  %2 = math.sqrt %1 : tensor<4xf64>
  return %0, %2 : tensor<4xf64>, tensor<4xf64>
}

// One op alone is no chain, so Herbie is not run on this function.
// CHECK-LABEL: func.func @single
// CHECK:         math.exp %arg0
func.func @single(%x: tensor<4xf64>) -> tensor<4xf64> {
  %0 = math.exp %x : tensor<4xf64>
  return %0 : tensor<4xf64>
}

// FIRST:      [poseidon] Herbie @log_one_plus: 1 FPCore(s) in 1 process(es)
// FIRST:      fake-herbie: (FPCore (x0) :name "log_one_plus/0" :precision binary64 (log (+ 1 x0)))
// FIRST:      Saved Herbie output to cache file {{.*}}/c/cachedHerbieOutput_default_
// FIRST:      [poseidon] Herbie @two_outputs: 2 FPCore(s) in 1 process(es)
// FIRST:      fake-herbie: (FPCore (x0) :name "two_outputs/0" :precision binary64 (exp x0))
// FIRST-NEXT: fake-herbie: (FPCore (x0) :name "two_outputs/1" :precision binary64 (sqrt (- (exp x0) 1)))
// FIRST-NOT:  Herbie @single

// Run again, Herbie is not: both answers are read from the cache.
// AGAIN:      Using cached Herbie output from {{.*}}/c/cachedHerbieOutput_default_
// AGAIN:      Using cached Herbie output from {{.*}}/c/cachedHerbieOutput_default_
// AGAIN-NOT:  fake-herbie:

// MISSING: Execution failed:
// MISSING: error: poseidon-rewrite: could not run Herbie ({{.*}}no-such-herbie)

// With herbie-results as well, the file answers and Herbie is not run: the
// last RUN line names a Herbie that does not exist.
