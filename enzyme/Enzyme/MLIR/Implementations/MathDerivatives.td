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
def : MLIRDerivative<"math", "AtanOp", (Op $x),
                    [
                      (CheckedMulF (DiffeRet), (DivF (ConstantFP<"1.0","arith","ConstantOp"> $x):$one, (AddF (MulF $x, $x), $one)))
                    ]
                  >;
def : MLIRDerivative<"math", "AbsFOp", (Op $x),
                    [
                      // TODO: handle complex
                      (Arith_Select (CmpF (Arith_OGE), $x, (ConstantFP<"0","arith","ConstantOp"> $x)), (DiffeRet), (NegF (DiffeRet)))
                    ]
                  >;
