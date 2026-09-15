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
def : MLIRDerivative<"arith", "MaxNumFOp", (Op $x, $y),
                    [
                      (Arith_Select (CmpF (Arith_OLT), $x, $y), (ConstantFP<"0","arith","ConstantOp"> $x), (DiffeRet)),
                      (Arith_Select (CmpF (Arith_OLT), $x, $y), (DiffeRet), (ConstantFP<"0","arith","ConstantOp"> $y)),
                    ]
                  >;

def IsPosZero : SubRoutine<(Op $x), (CmpF (Arith_OEQ),
                                          (CopySignF (ConstantFP<"1","arith","ConstantOp">), $x),
                                          (ConstantFP<"1","arith","ConstantOp">))>;
def IsNegZero : SubRoutine<(Op $x), (CmpF (Arith_OEQ),
                                          (CopySignF (ConstantFP<"1","arith","ConstantOp">), $x),
                                          (ConstantFP<"-1","arith","ConstantOp">))>;

def : MLIRDerivative<"arith", "MaximumFOp", (Op $x, $y),
                    [
                      (Arith_Select (CmpF (Arith_OLT), $x, $y),
                        (ConstantFP<"0","arith","ConstantOp"> $x),
                        (Arith_Select (AndI (IsNegZero $x), (IsPosZero $y)),
                            (ConstantFP<"0","arith","ConstantOp"> $x),
                            (DiffeRet))),
                      (Arith_Select (CmpF (Arith_OLT), $x, $y),
                        (DiffeRet),
                        (Arith_Select (AndI (IsNegZero $x), (IsPosZero $y)),
                            (DiffeRet),
                            (ConstantFP<"0","arith","ConstantOp"> $x)))
                    ]>;

def : MLIRDerivative<"arith", "MinimumFOp", (Op $x, $y),
                    [
                      (Arith_Select (CmpF (Arith_OLT), $x, $y),
                        (DiffeRet),
                        (Arith_Select (AndI (IsNegZero $x), (IsPosZero $y)),
                            (DiffeRet),
                            (ConstantFP<"0","arith","ConstantOp"> $x))),
                      (Arith_Select (CmpF (Arith_OLT), $x, $y),
                        (ConstantFP<"0","arith","ConstantOp"> $y),
                        (Arith_Select (AndI (IsNegZero $x), (IsPosZero $y)),
                            (ConstantFP<"0","arith","ConstantOp"> $y),
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
