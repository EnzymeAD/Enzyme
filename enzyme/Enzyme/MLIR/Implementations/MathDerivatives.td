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
