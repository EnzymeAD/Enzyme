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

def : MLIRDerivative<"complex", "SinOp", (Op $x),
                    [
                      (CheckedComplexMul (DiffeRet), (ComplexCos $x))
                    ]
                  >;

def : MLIRDerivative<"complex", "CosOp", (Op $x),
                    [
                      (CheckedComplexMul (DiffeRet), (ComplexNeg (ComplexSin $x)))
                    ]
                  >;

def : MLIRDerivative<"complex", "TanOp", (Op $x),
                    [
                      (CheckedComplexDiv
                        (DiffeRet),
                        (ComplexMul (ComplexCos $x):$c, $c)
                      )
                    ]
                  >;

def : MLIRDerivative<"complex", "Atan2Op", (Op $y, $x),
                    [
                      (CheckedComplexMul
                        (DiffeRet),
                        (ComplexDiv
                          $x,
                          (ComplexAdd (ComplexMul $x, $x), (ComplexMul $y, $y))
                        )
                      ),
                      (CheckedComplexMul
                        (DiffeRet),
                        (ComplexDiv
                          (ComplexNeg $y),
                          (ComplexAdd (ComplexMul $x, $x), (ComplexMul $y, $y))
                        )
                      )
                    ]
                  >;

def : MLIRDerivative<"complex", "TanhOp", (Op $x),
                    [
                      (CheckedComplexMul
                        (DiffeRet),
                        (ComplexSub
                          (ComplexConstant<"1.0"> $x),
                          (ComplexMul (ComplexTanh $x):$t, $t)
                        )
                      )
                    ]
                  >;

def : MLIRDerivative<"complex", "ExpOp", (Op $x),
                    [
                      (CheckedComplexMul (DiffeRet), (ComplexExp $x))
                    ]
                  >;

def : MLIRDerivative<"complex", "Expm1Op", (Op $x),
                    [
                      (CheckedComplexMul (DiffeRet), (ComplexExp $x))
                    ]
                  >;

def : MLIRDerivative<"complex", "RsqrtOp", (Op $x),
                    [
                      (CheckedComplexMul
                        (DiffeRet),
                        (ComplexDiv
                          (ComplexNeg (ComplexRsqrt $x)),
                          (ComplexMul (ComplexConstant<"2.0"> $x), $x)
                        )
                      )
                    ]
                  >;

def : MLIRDerivative<"complex", "SqrtOp", (Op $x),
                    [
                      (Arith_Select
                        (ComplexEq $x, (ComplexConstant<"0"> $x):$zero),
                        $zero,
                        (ComplexDiv
                          (DiffeRet),
                          (ComplexMul (ComplexConstant<"2.0"> $x), (ComplexSqrt $x))
                        )
                      )
                    ]
                  >;
