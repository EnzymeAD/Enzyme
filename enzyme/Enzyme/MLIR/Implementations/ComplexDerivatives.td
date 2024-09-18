include "Common.td"

def : MLIRDerivative<"complex", "AddOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (DiffeRet),
                    ]
                  >;

def CMul : ComplexInst<"MulOp">;

def : MLIRDerivative<"complex", "MulOp", (Op $x, $y),
                    [
                      (CMul (DiffeRet), $y),
                      (CMul (DiffeRet), $x)
                    ]
                  >;
