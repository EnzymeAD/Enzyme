include "Common.td"

class CirInst<string m> : Inst<m, "cir">;
class CirInactiveOp<string m> : InactiveOp<"cir", m>;

def CirEQ : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "cir::CmpOpKind::eq">;
def CirGE : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "cir::CmpOpKind::ge">;
def CirLT : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "cir::CmpOpKind::lt">;
def CirAnd : CirInst<"AndOp">;

// Some constant for de-duplication
class CirCF<string v> : ConstantFP<v,"cir","ConstantOp">;
def CirCF0 : CirCF<"0">;
def CirCF1 : CirCF<"1">;
def CirCF2 : CirCF<"2">;
def CirCF10 : CirCF<"10">;
def CirCFNeg1 : CirCF<"-1">;

def CirFAdd : CirInst<"FAddOp">;
def CirFSub : CirInst<"FSubOp">;
// TODO: Implement strong IEEE 754. For now, we skip this like LLVM IR.
def CirFMul : CirInst<"FMulOp">;
def CirFDiv : CirInst<"FDivOp">;
def CirFNeg : CirInst<"FNegOp">;
def CirSqrt : CirInst<"SqrtOp">;
def CirSin  : CirInst<"SinOp">;
def CirSinh : CirInst<"SinhOp">;
def CirCos  : CirInst<"CosOp">;
def CirCosh : CirInst<"CoshOp">;
def CirTan : CirInst<"TanOp">;
def CirTanh : CirInst<"TanhOp">;
def CirLog : CirInst<"LogOp">;
def CirExp  : CirInst<"ExpOp">;
def CirExp2  : CirInst<"Exp2Op">;
def CirExp10  : CirInst<"Exp10Op">;
def CirPow  : CirInst<"PowOp">;
def CirSelect  : CirInst<"SelectOp">;
def CirCmp  : CirInst<"CmpOp">;
def CirFAbs  : CirInst<"FAbsOp">;
def CirCopysign  : CirInst<"CopysignOp">;
def CirFloor: CirInst<"FloorOp">;

// Arithmetic
// TODO: Add asin, acos for FastMath
def : MLIRDerivative<"cir", "FAddOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (DiffeRet),
                    ]
                  >;

def : MLIRDerivative<"cir", "FSubOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (CirFNeg (DiffeRet)),
                    ]
                  >;

def : MLIRDerivative<"cir", "FNegOp", (Op $x),
                    [
                      (CirFNeg (DiffeRet)),
                    ]
                  >;

def : MLIRDerivative<"cir", "FMulOp", (Op $x, $y),
                    [
                      (CirFMul (DiffeRet), $y),
                      (CirFMul (DiffeRet), $x)
                    ]
                  >;

def : MLIRDerivative<"cir", "FDivOp", (Op $x, $y),
                    [
                      (CirFDiv (DiffeRet), $y),
                      (CirFNeg (CirFMul (CirFDiv (DiffeRet), $y), (CirFDiv $x, $y)))
                    ],
                    (CirFDiv
                      (CirFSub (SelectIfActive $x, (CirFMul (Shadow $x), $y), (CirCF0 (Shadow $x))),
                               (SelectIfActive $y, (CirFMul (Shadow $y), $x), (CirCF0 (Shadow $y)))),
                      (CirFMul $y, $y))
                  >;

def : MLIRDerivative<"cir", "FMulAddOp", (Op $x, $y, $z),
                    [
                      (CirFMul (DiffeRet), $y),
                      (CirFMul (DiffeRet), $x),
                      (DiffeRet),
                    ]>;

def : MLIRDerivative<"cir", "FMAOp", (Op $x, $y, $z),
                    [
                      (CirFMul (DiffeRet), $y),
                      (CirFMul (DiffeRet), $x),
                      (DiffeRet),
                    ]>;

def CirIsPosZero : SubRoutine<(Op $x), (CirCmp (CirEQ),
                                          (CirCopysign (TypeOf $x), (CirCF1 $x), $x),
                                          (CirCF1 $x))>;
def CirIsNegZero : SubRoutine<(Op $x), (CirCmp (CirEQ),
                                          (CirCopysign (TypeOf $x), (CirCF1 $x), $x),
                                          (CirCFNeg1 $x))>;

def : MLIRDerivative<"cir", "FMaxNumOp", (Op $x, $y),
                    [
                      (CirSelect (CirCmp (CirLT), $x, $y), (CirCF0 $x), (DiffeRet)),
                      (CirSelect (CirCmp (CirLT), $x, $y), (DiffeRet), (CirCF0 $y)),
                    ]>;

def : MLIRDerivative<"cir", "FMinNumOp", (Op $x, $y),
                    [
                      (CirSelect (CirCmp (CirLT), $x, $y), (DiffeRet), (CirCF0 $x)),
                      (CirSelect (CirCmp (CirLT), $x, $y), (CirCF0 $y), (DiffeRet)),
                    ]>;

def : MLIRDerivative<"cir", "FMaximumOp", (Op $x, $y),
                    [
                      (CirSelect (CirCmp (CirLT), $x, $y),
                        (CirCF0 $x),
                        (CirSelect (CirAnd (CirIsNegZero $x), (CirIsPosZero $y)),
                            (CirCF0 $x),
                            (DiffeRet))),
                      (CirSelect (CirCmp (CirLT), $x, $y),
                        (DiffeRet),
                        (CirSelect (CirAnd (CirIsNegZero $x), (CirIsPosZero $y)),
                            (DiffeRet),
                            (CirCF0 $x)))
                    ]>;

def : MLIRDerivative<"cir", "FMinimumOp", (Op $x, $y),
                    [
                      (CirSelect (CirCmp (CirLT), $x, $y),
                        (DiffeRet),
                        (CirSelect (CirAnd (CirIsNegZero $x), (CirIsPosZero $y)),
                            (DiffeRet),
                            (CirCF0 (Shadow $x)))),
                      (CirSelect (CirCmp (CirLT), $x, $y),
                        (CirCF0 (Shadow $y)),
                        (CirSelect (CirAnd (CirIsNegZero $x), (CirIsPosZero $y)),
                            (CirCF0 (Shadow $y)),
                            (DiffeRet)))
                    ]>;

// x - floor(x / y) * y
// d/dx = 1
// d/dy = -floor(x / y)
def : MLIRDerivative<"cir", "FModOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (CirFMul (DiffeRet),
                        (CirFNeg (CirCopysign (TypeOf $x), (CirFloor (CirFAbs (CirFDiv $x, $y):$div)), $div)))
                    ]
                  >;

def : MLIRDerivative<"cir", "FRemOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (CirFMul (DiffeRet),
                        (CirFNeg (CirCopysign (TypeOf $x), (CirFloor (CirFAbs (CirFDiv $x, $y):$div)), $div)))
                    ]
                  >;

def : MLIRDerivative<"cir", "SelectOp", (Op $pred, $x, $y),
                    [
                      (AssertingInactiveArg),
                      (CirSelect $pred, (DiffeRet), (CirCF0 $x)),
                      (CirSelect $pred, (CirCF0 $x), (DiffeRet)),
                    ]
                  >;

def : MLIRDerivative<"cir", "FAbsOp", (Op $x),
                    [
                      (CirSelect (CirCmp (CirGE), $x, (CirCF0 $x)), (DiffeRet), (CirFNeg (DiffeRet)))
                    ]
                  >;

// |x| * sign(y)
// d/dx = sign(x) * sign(y)
// d/dy = 0
def : MLIRDerivative<"cir", "CopysignOp", (Op $x, $y),
                    [
                      (CirFMul
                        (DiffeRet),
                        (CirFMul
						// Workaround for https://github.com/llvm/llvm-project/pull/226876
                          (CirCopysign (TypeOf $x), (CirCF1 $x), $x),
                          (CirCopysign (TypeOf $x), (CirCF1 $y), $y))),
                      (InactiveArg)
                    ]
                  >;

// Math
def : MLIRDerivative<"cir", "SinOp", (Op $x),
                    [
                      (CirFMul (DiffeRet), (CirCos $x)),
                    ]>;

def : MLIRDerivative<"cir", "SinhOp", (Op $x),
                    [
                      (CirFMul (DiffeRet), (CirCosh $x)),
                    ]>;

def : MLIRDerivative<"cir", "ASinOp", (Op $x),
                    [
                      (CirFDiv (DiffeRet), (CirSqrt (CirFSub (CirCF1 $x), (CirFMul $x, $x)))),
                    ]>;

def : MLIRDerivative<"cir", "CosOp", (Op $x),
                    [
                      (CirFMul (DiffeRet), (CirFNeg(CirSin $x))),
                    ]>;

def : MLIRDerivative<"cir", "CoshOp", (Op $x),
                    [
                      (CirFMul (DiffeRet), (CirSinh $x)),
                    ]>;

def : MLIRDerivative<"cir", "ACosOp", (Op $x),
                    [
                      (CirFNeg (CirFDiv (DiffeRet), (CirSqrt (CirFSub (CirCF1 $x), (CirFMul $x, $x))))),
                    ]>;

def : MLIRDerivative<"cir", "TanOp", (Op $x),
                    [
                      (CirFMul (DiffeRet), (CirFAdd (CirCF1 $x), (CirFMul (CirTan $x), (CirTan $x)))),
                    ]>;

def : MLIRDerivative<"cir", "TanhOp", (Op $x),
                    [
                      (CirFMul (DiffeRet), (CirFSub (CirCF1 $x), (CirFMul (CirTanh $x), (CirTanh $x)))),
                    ]>;

def : MLIRDerivative<"cir", "ATanOp", (Op $x),
                    [
                      (CirFDiv (DiffeRet), (CirFAdd (CirCF1 $x), (CirFMul $x, $x)))
                    ]
                  >;

def : MLIRDerivative<"cir", "ATan2Op", (Op $y, $x),
                    [
                      (CirFMul (DiffeRet), (CirFDiv $x, (CirFAdd (CirFMul $x, $x), (CirFMul $y, $y)))),
                      (CirFMul (DiffeRet), (CirFNeg (CirFDiv $y, (CirFAdd (CirFMul $x, $x), (CirFMul $y, $y)))))
                    ]
                  >;

def : MLIRDerivative<"cir", "ExpOp", (Op $x),
                    [
                      (CirFMul (DiffeRet), (CirExp $x)),
                    ]>;

def : MLIRDerivative<"cir", "Exp2Op", (Op $x),
                    [
                      (CirFMul
                        (DiffeRet),
                        (CirFMul
                          (CirExp2 $x),
                          (CirLog
                            (CirCF2 $x)
                            )))
                    ]
                  >;

def : MLIRDerivative<"cir", "Exp10Op", (Op $x),
                    [
                      (CirFMul (DiffeRet), (CirFMul (CirExp10 $x), (CirLog (CirCF10 $x))))
                    ]>;

def : MLIRDerivative<"cir", "SqrtOp", (Op $x),
                    [
                      (CirSelect (CirCmp (CirEQ), $x, (CirCF0 $x):$zero), $zero, (CirFDiv (DiffeRet), (CirFMul (CirCF2 $x), (CirSqrt $x))))
                    ]
                  >;

def : MLIRDerivative<"cir", "PowOp", (Op $x, $y),
                    [
                      (CirFMul (DiffeRet), (CirFMul $y, (CirPow (TypeOf $x), $x, (CirFSub $y, (CirCF1 $y))))),
                      (CirFMul (DiffeRet), (CirSelect (CirCmp (CirEQ), $x, (CirCF0 $x):$pzero), $pzero, (CirFMul (CirPow (TypeOf $x), $x, $y), (CirLog $x))))
                    ]
                  >;

def : MLIRDerivative<"cir", "LogOp", (Op $x),
                    [
                      (CirFDiv (DiffeRet), $x)
                    ]
                  >;

def : MLIRDerivative<"cir", "Log2Op", (Op $x),
                    [
                      (CirFDiv
                        (DiffeRet),
                        (CirFMul
                          $x,
                          (CirLog (CirCF2 $x))))
                    ]
                  >;

def : MLIRDerivative<"cir", "Log10Op", (Op $x),
                    [
                      (CirFDiv
                        (DiffeRet),
                        (CirFMul
                          $x,
                          (CirLog (CirCF10 $x))))
                    ]
                  >;


// TODO: Add cir.complex member

// Memory
// TODO: Cir Cast cannot be single op as they aggregate different conversation into 1 cir.cast
def : MemoryIdentityOp<"cir", "StoreOp", [1], [0]>;
def : ReadOnlyIdentityOp<"cir", "LoadOp", [0]>;
def : AllocationOp<"cir", "AllocaOp">;

def : ReadOnlyIdentityOp<"cir", "GetMemberOp", [0]>;
def : ReadOnlyIdentityOp<"cir", "GetElementOp", [0]>;
def : ReadOnlyIdentityOp<"cir", "PtrStrideOp", [0]>;
def : ReadOnlyIdentityOp<"cir", "BaseClassAddrOp", [0]>;
def : ReadOnlyIdentityOp<"cir", "DerivedClassAddrOp", [0]>;
def : ReadOnlyIdentityOp<"cir", "ComplexRealPtrOp", [0]>;
def : ReadOnlyIdentityOp<"cir", "ComplexImagPtrOp", [0]>;
def : ReadOnlyIdentityOp<"cir", "PtrMaskOp", [0]>;
def : ReadOnlyIdentityOp<"cir", "GetRuntimeMemberOp", [0]>;
def : MemoryIdentityOp<"cir", "ExtractMemberOp", [0], [0]>;
def : MemoryIdentityOp<"cir", "InsertMemberOp", [0, 1], [0, 1]>;
def : MemoryIdentityOp<"cir", "VecExtractOp", [0], [0]>;
def : MemoryIdentityOp<"cir", "VecInsertOp", [0, 1], [0, 1]>;
def : CirInactiveOp<"ObjSizeOp">;
def : CirInactiveOp<"PtrDiffOp">;
def : CirInactiveOp<"StackSaveOp">;
def : CirInactiveOp<"StackRestoreOp">;
def : CirInactiveOp<"LifetimeStartOp">;
def : CirInactiveOp<"LifetimeEndOp">;
def : CirInactiveOp<"VTableAddrPointOp">;
def : CirInactiveOp<"VTableGetVPtrOp">;

// Control flow
def : ReturnOp<"cir", "ReturnOp">;
def : CallOp<"cir", "CallOp">;
def : BranchOp<"cir", "BrOp">;
def : BranchOp<"cir", "BrCondOp">;
def : BranchOp<"cir", "SwitchFlatOp">;

def : CirInactiveOp<"ConstantOp">;
def : CirInactiveOp<"CeilOp">;