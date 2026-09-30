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
                    // (dx - dy * (x / y)) / y rather than (dx * y - dy * x) / (y * y), see the
                    // FDiv rule in InstructionDerivatives.td.
                    (CheckedDivF (SubF (SelectIfActive $x, (Shadow $x), (ConstantFP<"0","arith", "ConstantOp"> $x)), (SelectIfActive $y, (MulF (Shadow $y), (DivF $x, $y)), (ConstantFP<"0","arith","ConstantOp"> $y))), $y)
                  >;
def : MLIRDerivative<"arith", "MaxNumFOp", (Op $x, $y),
                    [
                      (Arith_Select (CmpF (Arith_OLT), $x, $y), (ConstantFP<"0","arith","ConstantOp"> (Shadow $x)), (DiffeRet)),
                      (Arith_Select (CmpF (Arith_OLT), $x, $y), (DiffeRet), (ConstantFP<"0","arith","ConstantOp"> (Shadow $y))),
                    ]
                  >;

def IsPosZero : SubRoutine<(Op $x), (CmpF (Arith_OEQ),
                                          (CopySignF (ConstantFP<"1","arith","ConstantOp"> $x), $x),
                                          (ConstantFP<"1","arith","ConstantOp"> $x))>;
def IsNegZero : SubRoutine<(Op $x), (CmpF (Arith_OEQ),
                                          (CopySignF (ConstantFP<"1","arith","ConstantOp"> $x), $x),
                                          (ConstantFP<"-1","arith","ConstantOp"> $x))>;

def : MLIRDerivative<"arith", "MaximumFOp", (Op $x, $y),
                    [
                      (Arith_Select (CmpF (Arith_OLT), $x, $y),
                        (ConstantFP<"0","arith","ConstantOp"> (Shadow $x)),
                        (Arith_Select (AndI (IsNegZero $x), (IsPosZero $y)),
                            (ConstantFP<"0","arith","ConstantOp"> (Shadow $x)),
                            (DiffeRet))),
                      (Arith_Select (CmpF (Arith_OLT), $x, $y),
                        (DiffeRet),
                        (Arith_Select (AndI (IsNegZero $x), (IsPosZero $y)),
                            (DiffeRet),
                            (ConstantFP<"0","arith","ConstantOp"> (Shadow $x))))
                    ]>;

def : MLIRDerivative<"arith", "MinimumFOp", (Op $x, $y),
                    [
                      (Arith_Select (CmpF (Arith_OLT), $x, $y),
                        (DiffeRet),
                        (Arith_Select (AndI (IsNegZero $x), (IsPosZero $y)),
                            (DiffeRet),
                            (ConstantFP<"0","arith","ConstantOp"> (Shadow $x)))),
                      (Arith_Select (CmpF (Arith_OLT), $x, $y),
                        (ConstantFP<"0","arith","ConstantOp"> (Shadow $y)),
                        (Arith_Select (AndI (IsNegZero $x), (IsPosZero $y)),
                            (ConstantFP<"0","arith","ConstantOp"> (Shadow $y)),
                            (DiffeRet)))
                    ]>;

def : MLIRDerivative<"arith", "RemFOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (CheckedMulF (DiffeRet),
                        (NegF (CopySignF (FloorF (AbsF (DivF $x, $y):$div)), $div)))
                    ]
                  >;

def ExtF : ArithInst<"ExtFOp">;
def TruncF : ArithInst<"TruncFOp">;

def : ReadOnlyIdentityOp<"arith", "TruncFOp", [0], (Op $x), [(ExtF (TypeOf $x), (DiffeRet))]>;
def : ReadOnlyIdentityOp<"arith", "ExtFOp", [0], (Op $x), [(TruncF (TypeOf $x), (DiffeRet))]>;
