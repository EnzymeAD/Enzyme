include "Common.td"

def : MLIRDerivative<"math", "CosOp", (Op $x),
                    [
                      (CheckedMulF (DiffeRet), (NegF (SinF $x)))
                    ]
                  >;
def : MLIRDerivative<"math", "ExpOp", (Op $x),
                    [
                      (CheckedMulF (DiffeRet), (ExpF $x))
                    ]
                  >;
def : MLIRDerivative<"math", "FmaOp", (Op $x, $y, $z),
                    [
                      (CheckedMulF (DiffeRet), $y),
                      (CheckedMulF (DiffeRet), $x),
                      (DiffeRet)
                    ]
                  >;
def : MLIRDerivative<"math", "SinOp", (Op $x),
                    [
                      (CheckedMulF (DiffeRet), (CosF $x))
                    ]
                  >;
def : MLIRDerivative<"math", "SqrtOp", (Op $x),
                    [
                      (Arith_Select (CmpF (Arith_OEQ), $x, (ConstantFP<"0","arith","ConstantOp"> $x):$zero), $zero, (DivF (DiffeRet), (MulF (ConstantFP<"2.0","arith","ConstantOp"> $x), (SqrtF $x))))
                    ]
                  >;
def : MLIRDerivative<"math", "PowFOp", (Op $x, $y),
                    [
                      // d/dx pow(x, y) = y * pow(x, y - 1)
                      (CheckedMulF (DiffeRet), (MulF $y, (PowF $x, (SubF $y, (ConstantFP<"1.0","arith","ConstantOp"> $y))))),
                      // d/dy pow(x, y) = pow(x, y) * log(x), which the x == 0
                      // guard pins to zero where the log has none.
                      (CheckedMulF (DiffeRet), (Arith_Select (CmpF (Arith_OEQ), $x, (ConstantFP<"0","arith","ConstantOp"> $x):$pzero), $pzero, (MulF (PowF $x, $y), (LogF $x))))
                    ]
                  >;
def : MLIRDerivative<"math", "LogOp", (Op $x),
                    [
                      (CheckedDivF (DiffeRet), $x)
                    ]
                  >;
def : MLIRDerivative<"math", "AtanOp", (Op $x),
                    [
                      (CheckedMulF (DiffeRet), (DivF (ConstantFP<"1.0","arith","ConstantOp">):$one, (AddF (MulF $x, $x), $one)))
                    ]
                  >;
def : MLIRDerivative<"math", "AbsFOp", (Op $x),
                    [
                      (SelectIfComplex $x,
                        (ComplexCreate
                          (TypeOf $x),
                          (CheckedMulF (DiffeRet), (DivF (ComplexRe $x), (AbsF $x))),
                          (CheckedMulF (DiffeRet), (NegF (DivF (ComplexIm $x), (AbsF $x))))
                        ),
                        (Arith_Select (CmpF (Arith_OGE), $x, (ConstantFP<"0","arith","ConstantOp"> $x)), (DiffeRet), (NegF (DiffeRet)))
                      )
                    ]
                  >;
