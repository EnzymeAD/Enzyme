include "Common.td"

class MathInactiveOp<string m> : InactiveOp<"math", m>;

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
def : MLIRDerivative<"math", "Exp2Op", (Op $x),
                    [
                      (CheckedMulF
                        (DiffeRet),
                        (MulF
                          (Exp2F $x),
                          (LogF
                            (ConstantFP<"2","arith","ConstantOp"> $x)
                            )))
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
def : MLIRDerivative<"math", "Log2Op", (Op $x),
                    [
                      (CheckedDivF
                        (DiffeRet),
                        (MulF
                          $x,
                          (LogF (ConstantFP<"2","arith", "ConstantOp"> $x))))
                    ]
                  >;
def : MLIRDerivative<"math", "AtanOp", (Op $x),
                    [
                      (CheckedMulF (DiffeRet), (DivF (ConstantFP<"1.0","arith","ConstantOp"> $x):$one, (AddF (MulF $x, $x), $one)))
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

def : MLIRDerivative<"math", "Log1pOp", (Op $x),
                    [
                      (CheckedDivF
                        (DiffeRet),
                        (AddF $x, (ConstantFP<"1","arith","ConstantOp"> $x)))
                    ]
                  >;

def : MLIRDerivative<"math", "CopySignOp", (Op $x, $y),
                    [
                      (CheckedMulF
                        (DiffeRet),
                        (MulF
                          (CopySignF (ConstantFP<"1","arith","ConstantOp"> $x), $x),
                          (CopySignF (ConstantFP<"1","arith","ConstantOp"> $y), $y))),
                      (InactiveArg)
                    ]
                  >;

def : MLIRDerivative<"math", "CoshOp", (Op $x),
                    [
                      (CheckedMulF (DiffeRet), (SinhF $x))
                    ]
                  >;

def : MLIRDerivative<"math", "SinhOp", (Op $x),
                    [
                      (CheckedMulF (DiffeRet), (CoshF $x))
                    ]
                  >;

def : MLIRDerivative<"math", "TanhOp", (Op $x),
                    [
                      (CheckedDivF (DiffeRet), (MulF(CoshF $x):$c, $c))
                    ]
                  >;


def : MLIRDerivative<"math", "TanOp", (Op $x),
                    [
                      (CheckedDivF (DiffeRet), (MulF(CosF $x):$c, $c))
                    ]
                  >;

def : MLIRDerivative<"math", "AsinOp", (Op $x),
                    [
                      (CheckedDivF (DiffeRet), 
                        (SqrtF( SubF (ConstantFP<"1","arith","ConstantOp"> $x), (MulF $x, $x))))
                    ]
                  >;

def : MLIRDerivative<"math", "AcosOp", (Op $x),
                    [
                      (CheckedDivF (NegF (DiffeRet)), 
                        (SqrtF( SubF (ConstantFP<"1","arith","ConstantOp"> $x), (MulF $x, $x))))
                    ]
                  >;

def : MLIRDerivative<"math", "AsinhOp", (Op $x),
                    [
                      (CheckedDivF (DiffeRet), 
                        (SqrtF( AddF (ConstantFP<"1","arith","ConstantOp"> $x), (MulF $x, $x))))
                    ]
                  >;

def : MLIRDerivative<"math", "AcoshOp", (Op $x),
                    [
                      (CheckedDivF (DiffeRet), 
                        (SqrtF( SubF (MulF $x, $x), (ConstantFP<"1","arith","ConstantOp"> $x))))
                    ]
                  >;

def : MLIRDerivative<"math", "AtanhOp", (Op $x),
                    [
                      (CheckedDivF (DiffeRet), 
                        (SubF(ConstantFP<"1","arith","ConstantOp"> $x), (MulF $x, $x)))
                    ]
                  >;

def : MLIRDerivative<"math", "ExpM1Op", (Op $x),
                    [
                      (CheckedMulF (DiffeRet), (ExpF $x))
                    ]
                  >;

def : MLIRDerivative<"math", "Log10Op", (Op $x),
                    [
                      (CheckedDivF (DiffeRet), 
                        (MulF $x, (ConstantFP<"2.302585092994046","arith","ConstantOp"> $x)))
                    ]
                  >;

// d/dx rsqrt(x) = -1/2 · x^(-3/2) = -rsqrt(x) / (2x)
def : MLIRDerivative<"math", "RsqrtOp", (Op $x),
                    [
                      (CheckedMulF (DiffeRet),
                        (DivF (NegF (RsqrtF $x)),
                              (MulF (ConstantFP<"2","arith","ConstantOp"> $x), $x)))
                    ]
                  >;

def : MLIRDerivative<"math", "CbrtOp", (Op $x),
                    [
                      (CheckedDivF (CheckedMulF (DiffeRet), (CbrtF $x)),
                                   (MulF (ConstantFP<"3.0","arith","ConstantOp"> $x), $x))
                    ]
                  >;
def : MathInactiveOp<"CeilOp">;
def : MathInactiveOp<"FloorOp">;
