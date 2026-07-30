// RUN: %eopt -lower-enzyme-binomial-progress %s | FileCheck %s
// RUN: %eopt -lower-enzyme-binomial-progress -canonicalize %s | FileCheck %s --check-prefix=FOLD

// enzyme.binomial_progress returns the Revolve *advance distance*: with
// beta(s,t) = C(s+t,t) and t minimal such that beta(budget,t) >= num_steps, the
// midpoint of [num_steps - beta(budget-1,t), beta(budget,t-1)] clamped to
// [1, num_steps-1]. It grows like num_steps, not like num_steps^(1/budget).

// Constant operands fold to a plain constant. For (9, 3): t = 2, since
// beta(3,1) = 4 < 9 <= beta(3,2) = 10; the window is
// [9 - beta(2,2), beta(3,1)] = [9 - 6, 4] = [3, 4], midpoint 3.
func.func @cst() -> index {
  %n = arith.constant 9 : index
  %s = arith.constant 3 : index
  %r = enzyme.binomial_progress %n, %s : index
  return %r : index
}

// FOLD-LABEL: func.func @cst() -> index {
// FOLD-NEXT:    %[[R:.+]] = arith.constant 3 : index
// FOLD-NEXT:    return %[[R]] : index
// FOLD-NOT:     enzyme.binomial_progress

// A budget of 1 advances the whole remaining stretch: with one checkpoint left
// it is replayed from there. This is also what makes the per-slot advances sum
// to exactly the trip count across `budget` slots, which is what lets a driver
// that iterates once per checkpoint still reach the end of the primal.
func.func @budget_one() -> index {
  %n = arith.constant 40 : index
  %s = arith.constant 1 : index
  %r = enzyme.binomial_progress %n, %s : index
  return %r : index
}

// FOLD-LABEL: func.func @budget_one() -> index {
// FOLD-NEXT:    %[[R:.+]] = arith.constant 40 : index
// FOLD-NEXT:    return %[[R]] : index

// A single remaining step advances by one, never zero: a zero advance would
// leave the caller's loop without progress.
func.func @one_step() -> index {
  %n = arith.constant 1 : index
  %s = arith.constant 4 : index
  %r = enzyme.binomial_progress %n, %s : index
  return %r : index
}

// FOLD-LABEL: func.func @one_step() -> index {
// FOLD-NEXT:    %[[R:.+]] = arith.constant 1 : index
// FOLD-NEXT:    return %[[R]] : index

// The advance is a constant fraction of the interval, not an inverse-binomial
// index: (400, 4) is in the hundreds. It was 11 when this op returned the
// repetition count, which left a 362-step stretch with no interior checkpoint
// and made the reverse pass quadratic.
func.func @large() -> index {
  %n = arith.constant 400 : index
  %s = arith.constant 4 : index
  %r = enzyme.binomial_progress %n, %s : index
  return %r : index
}

// FOLD-LABEL: func.func @large() -> index {
// FOLD-NEXT:    %[[R:.+]] = arith.constant 282 : index
// FOLD-NEXT:    return %[[R]] : index

// Dynamic operands lower to the Revolve computation on scf/arith. The guard is
// a branch rather than a select because with budget <= 1 the loop below would
// leave beta at 1 and never terminate.
func.func @dyn(%n: index, %s: index) -> index {
  %r = enzyme.binomial_progress %n, %s : index
  return %r : index
}

// CHECK-LABEL: func.func @dyn(
// CHECK-SAME:    %[[N:.+]]: index, %[[S:.+]]: index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK:         %[[NSM:.+]] = arith.cmpi sle, %[[N]], %[[C1]]
// CHECK-NEXT:    %[[SSM:.+]] = arith.cmpi sle, %[[S]], %[[C1]]
// CHECK-NEXT:    %[[G:.+]] = arith.ori %[[NSM]], %[[SSM]]
// CHECK-NEXT:    %{{.+}} = scf.if %[[G]] -> (index) {
// CHECK-NEXT:      scf.yield %[[N]] : index
// CHECK-NEXT:    } else {
// beta = C(s+t, t), stepped from C(s+t-1, t-1) by *(s+t)/t.
// CHECK-NEXT:      %[[W:.+]]:2 = scf.while (%[[T:.+]] = %[[C0]], %[[B:.+]] = %[[C1]])
// CHECK-NEXT:        arith.cmpi slt, %[[B]], %[[N]]
// CHECK-NEXT:        scf.condition
// CHECK:           ^bb0(%[[T2:.+]]: index, %[[B2:.+]]: index):
// CHECK-NEXT:        %[[TN:.+]] = arith.addi %[[T2]], %[[C1]]
// CHECK-NEXT:        %[[SPT:.+]] = arith.addi %[[S]], %[[TN]]
// CHECK-NEXT:        %[[MUL:.+]] = arith.muli %[[B2]], %[[SPT]]
// CHECK-NEXT:        %[[DIV:.+]] = arith.divui %[[MUL]], %[[TN]]
// CHECK-NEXT:        scf.yield %[[TN]], %[[DIV]]
// Window edges n - beta(s-1,t) and beta(s,t-1), clamped, then the midpoint.
// CHECK:           %[[SUM:.+]] = arith.addi %[[S]], %[[W]]#0
// CHECK-NEXT:      %[[LN:.+]] = arith.muli %[[W]]#1, %[[S]]
// CHECK-NEXT:      %[[LD:.+]] = arith.divui %[[LN]], %[[SUM]]
// CHECK-NEXT:      %[[LO:.+]] = arith.subi %[[N]], %[[LD]]
// CHECK-NEXT:      %[[HN:.+]] = arith.muli %[[W]]#1, %[[W]]#0
// CHECK-NEXT:      %[[HI:.+]] = arith.divui %[[HN]], %[[SUM]]
// CHECK-NEXT:      %[[NM1:.+]] = arith.subi %[[N]], %[[C1]]
// CHECK-NEXT:      %[[CLO:.+]] = arith.maxsi %[[LO]], %[[C1]]
// CHECK-NEXT:      %[[CHI:.+]] = arith.minsi %[[HI]], %[[NM1]]
// CHECK-NEXT:      %[[C2:.+]] = arith.constant 2 : index
// CHECK-NEXT:      %[[ADD:.+]] = arith.addi %[[CLO]], %[[CHI]]
// CHECK-NEXT:      %[[MID:.+]] = arith.divui %[[ADD]], %[[C2]]
// Cap so one step is left for each of the s-1 slots still to be placed,
// otherwise the advances can exhaust the interval before the slots run out.
// CHECK-NEXT:      %[[SM1:.+]] = arith.subi %[[S]], %[[C1]]
// CHECK-NEXT:      %[[CAP:.+]] = arith.subi %[[N]], %[[SM1]]
// CHECK-NEXT:      %[[CAPPED:.+]] = arith.minsi %[[MID]], %[[CAP]]
// CHECK-NEXT:      %[[RES:.+]] = arith.maxsi %[[CAPPED]], %[[C1]]
// CHECK-NEXT:      scf.yield %[[RES]] : index
// CHECK-NOT:     enzyme.binomial_progress

// Dynamic i64 operands lower the same way, using i64 constants.
func.func @dyn_i64(%n: i64, %s: i64) -> i64 {
  %r = enzyme.binomial_progress %n, %s : i64
  return %r : i64
}

// CHECK-LABEL: func.func @dyn_i64(
// CHECK:         %[[ONE64:.+]] = arith.constant 1 : i64
// CHECK:         scf.if
// CHECK:           scf.while
// CHECK:             arith.cmpi slt
// CHECK:             scf.condition
// CHECK:             arith.muli
// CHECK:             arith.divui
// CHECK:           arith.maxsi
// CHECK:           arith.minsi
// CHECK-NOT:     enzyme.binomial_progress

// Unranked tensor operands are left untouched by this pass; tensor lowering
// is handled elsewhere.
func.func @tensor(%n: tensor<*xi64>, %s: tensor<*xi64>) -> tensor<*xi64> {
  %r = enzyme.binomial_progress %n, %s : tensor<*xi64>
  return %r : tensor<*xi64>
}

// CHECK-LABEL: func.func @tensor(
// CHECK:         enzyme.binomial_progress
