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

def : MLIRDerivative<"complex", "ImOp", (Op $x),
                    [
                      (ComplexCreate
                        (TypeOf $x),
                        (ConstantFP<"0", "arith", "ConstantOp">),
                        (NegF (DiffeRet))
                      )
                    ],
                    (ComplexIm (Shadow $x))
                  >;

def : MLIRDerivative<"complex", "ReOp", (Op $x),
                    [
                      (ComplexCreate
                        (TypeOf $x),
                        (DiffeRet),
                        (ConstantFP<"0", "arith", "ConstantOp">)
                      )
                    ],
                    (ComplexRe (Shadow $x))
                  >;

def : MLIRDerivative<"complex", "CreateOp", (Op $re, $im),
                    [
                      (ComplexRe (DiffeRet)),
                      (ComplexIm (DiffeRet))
                    ]
                  >;

