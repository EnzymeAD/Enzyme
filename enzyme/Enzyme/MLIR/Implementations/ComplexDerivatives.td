include "Common.td"

def : MLIRDerivative<"complex", "AddOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (DiffeRet),
                    ]
                  >;

def : MLIRDerivative<"complex", "MulOp", (Op $x, $y),
                    [
                      (CheckedComplexMul (DiffeRet), $y),
                      (CheckedComplexMul (DiffeRet), $x)
                    ]
                  >;

def : MLIRDerivative<"complex", "DivOp", (Op $x, $y),
                    [
                      (CheckedComplexDiv (DiffeRet), $y),
                      (ComplexNeg
                        (ComplexMul
                          (CheckedComplexDiv (DiffeRet), $y),
                          (ComplexDiv $x, $y)
                        )
                      )
                    ],
                    (CheckedComplexDiv
                      (ComplexSub
                        (SelectIfActive $x,
                          (ComplexMul (Shadow $x), $y),
                          (ComplexConstant<"0"> $x)
                        ),
                        (SelectIfActive $y,
                          (ComplexMul (Shadow $y), $x),
                          (ComplexConstant<"0"> $y)
                        )
                      ),
                      (ComplexMul $y, $y)
                    )
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
                      (NegF (ComplexIm (DiffeRet)))
                    ]
                  >;

