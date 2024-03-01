include "Common.td"

def : MLIRDerivative<"arith", "AddFOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (DiffeRet),
                    ]
                  >;

def : MLIRDerivative<"arith", "SubFOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (NegF (DiffeRet)),
                    ]
                  >;
def : MLIRDerivative<"arith", "NegFOp", (Op $x),
                    [
                      (NegF (DiffeRet)),
                    ]
                  >;
def : MLIRDerivative<"arith", "MulFOp", (Op $x, $y),
                    [
                      (CheckedMulF (DiffeRet), $y),
                      (CheckedMulF (DiffeRet), $x)
                    ]
                  >;
def : MLIRDerivative<"arith", "DivFOp", (Op $x, $y), 
                    [
                      (CheckedDivF (DiffeRet), $y),
                      (NegF (MulF (CheckedDivF (DiffeRet), $y), (DivF $x, $y)))
                    ],
                    (CheckedDivF (SubF (SelectIfActive $x, (MulF (Shadow $x), $y), (ConstantFP<"0","arith", "ConstantOp"> $x)), (SelectIfActive $y, (MulF (Shadow $y), $x), (ConstantFP<"0","arith","ConstantOp"> $y))), (MulF $y, $y))
                  >;

def ExtF : ArithInst<"ExtFOp">;
def TruncF : ArithInst<"TruncFOp">;

def : ReadOnlyIdentityOp<"arith", "TruncFOp", [0], (Op $x), [(ExtF (TypeOf $x), (DiffeRet))]>;
def : ReadOnlyIdentityOp<"arith", "ExtFOp", [0], (Op $x), [(TruncF (TypeOf $x), (DiffeRet))]>;
