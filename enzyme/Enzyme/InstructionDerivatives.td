class Type;
class ForwardFromSummedReverse {
  
}

class CallPattern<dag patternToMatch, list<string> funcNames, list<dag> resultOps, dag forwardOps> {
  dag PatternToMatch = patternToMatch;
  list<string> names = funcNames;
  list<dag> ArgDerivatives = resultOps;
  dag ArgDuals = forwardOps;
}

class IntrPattern<dag patternToMatch, list<list<string>> funcNames, list<dag> resultOps, dag forwardOps> {
  dag PatternToMatch = patternToMatch;
  list<list<string>> names = funcNames;
  list<dag> ArgDerivatives = resultOps;
  dag ArgDuals = forwardOps;
}

class Inst<string mnemonic> {
  string name = mnemonic;
}

class Attribute<string name_> {
  string name = name_;
}

// So we can also extend it by not input related 
// types, eg. for mem related Intrinsics
class TypeInfo {
}

class TypeOf : TypeInfo {
}
class VectorSize : TypeInfo {
}

def ReadNone : Attribute<"ReadNone">;
def ReadOnly : Attribute<"ReadOnly">;
def NoUnwind : Attribute<"NoUnwind">;


def ShadowOrZero {
  
}

// Define ops to rewrite.

def CheckedMul : Inst<"CheckedMul">;
def CheckedDiv : Inst<"CheckedDiv">;
def InsertElement : Inst<"InsertElement">;
def ShuffleVector : Inst<"ShuffleVector">;
def VectorSplat : Inst<"VectorSplat">;
def SIToFP : Inst<"SIToFP">;
def Or : Inst<"Or">;
def And : Inst<"And">;
def IAdd : Inst<"Add">;
def ISub : Inst<"Sub">;
def FDiv : Inst<"FDiv">;
def FAdd : Inst<"FAdd">;
def FSub : Inst<"FSub">;
def FMul : Inst<"FMul">;
def FNeg : Inst<"FNeg">;
def Select : Inst<"Select">;
def ICmpEQ : Inst<"ICmpEQ">;
def FCmpUEQ : Inst<"FCmpUEQ">;
def FCmpOEQ : Inst<"FCmpOEQ">;
def FCmpOLT : Inst<"FCmpOLT">;

class Call<dag mnemonic, list<Attribute> _fnattrs=[]> : Inst<"Call"> {
  dag func = mnemonic;
  list<Attribute> fnattrs = _fnattrs;
}

// Types is the list of indices of which arguments whose types
// are used in the intrinsic definition. As most commonly, this is
// simply the 0'th arg, that is the default.
class Intrinsic<string name_, list<int> types_ = [0]> : Inst<"Intrinsic"> {
  string name = name_;
  list<int> types = types_;
}


def Op {
}

class SubRoutine<dag patternToMatch, dag insts_> {
  dag PatternToMatch = patternToMatch;
  dag insts = insts_;
}

def FExp : Intrinsic<"exp">;
def FCos : Intrinsic<"cos">;
def FSin : Intrinsic<"sin">;

class ExtractValue<list<int> indices_> : Inst<"ExtractValue"> {
  list<int> indices = indices_;
}

class InsertValue<list<int> indices_> : Inst<"InsertValue"> {
  list<int> indices = indices_;
}


def ArrayRet {
}

def CFMul : SubRoutine<(Op (Op $re1, $im1):$z1, (Op $re2, $im2):$z2),
                  (ArrayRet
                    (FSub (FMul $re1, $re2), (FMul $im1, $im2)),
                    (FAdd (FMul $re1, $im2), (FMul $re2, $im1))
                  )>;
def CFNeg : SubRoutine<(Op (Op $re, $im):$z),
    (ArrayRet
      (FNeg $re),
      (FNeg $im)
    )>;
def CFExp : SubRoutine<(Op (Op $re, $im):$z),
    (ArrayRet
      (FMul (FExp $re):$exp, (FCos $im)),
      (FMul $exp, (FSin $im))
    )>;

def DifR : Inst<"DifR">;

// Same function as the one being called
def SameFunc {
}

// Same types / attributes / callingconv as original
// Different name, however
class SameTypesFunc<string name_> {
  string name = name_;
}

// Same attributes / callingconv as original
// Different name, however.
// Same types, with the new ones added
class PrependArgTypesFunc<string name_, list<string> pretys_> {
  string name = name_;
  list<string> pretys = pretys_;
}

def InactiveArg {
  
}

class GlobalExpr<string val> {
  string value = val;
}
class ConstantFP<string val> {
  string value = val;
}
class ConstantCFP<string rval, string ival> {
  string rvalue = rval;
  string ivalue = ival;
}
class ConstantInt<int value_, int bitwidth_=0> {
  int value = value_;
  int bitwidth=bitwidth_;
}
class Undef {
}

class DiffeRet<list<int> indices_ = [-1]> {
  list<int> indices = indices_;
}

def Shadow {
}

def : CallPattern<(Op $x),
                  ["atan", "atanf", "atanl", "__fd_atan_1"],
                  [(CheckedDiv (DiffeRet<>), (FAdd (FMul $x, $x), (ConstantFP<"1.0"> $x)))],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $y, $x),
                  ["atan2", "atan2f", "atan2l"],
                  [
                  (FDiv (FMul (DiffeRet<>), $x), (FAdd (FMul $x, $x), (FMul $y, $y))),
                  (FNeg (FDiv (FMul (DiffeRet<>), $y), (FAdd (FMul $x, $x), (FMul $y, $y))))
                  ],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x),
                  ["cbrt", "cbrtf", "cbrtl"],
                  [(FDiv (FMul (DiffeRet<>), (Call<(SameFunc), [ReadNone,NoUnwind]> $x) ), (FMul (ConstantFP<"3.0"> $x), $x))],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x, $y),
                  ["hypot", "hypotf", "hypotl"],
                  [
                    (FDiv (FMul (DiffeRet<>), $x), (Call<(SameFunc), [ReadNone,NoUnwind]> $x, $y)),
                    (FDiv (FMul (DiffeRet<>), $y), (Call<(SameFunc), [ReadNone,NoUnwind]> $x, $y))
                  ],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["tanh"],
                  [(FDiv (DiffeRet<>), (FMul(Call<(SameTypesFunc<"cosh">), [ReadNone,NoUnwind]> $x):$c, $c))],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["tanhf"],
                  [(FDiv (DiffeRet<>), (FMul(Call<(SameTypesFunc<"coshf">), [ReadNone,NoUnwind]> $x):$c, $c))],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["cosh"],
                  [(FMul (DiffeRet<>), (Call<(SameTypesFunc<"sinh">), [ReadNone,NoUnwind]> $x))],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x),
                  ["coshf"],
                  [(FMul (DiffeRet<>), (Call<(SameTypesFunc<"sinhf">), [ReadNone,NoUnwind]> $x))],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["sinh"],
                  [(FMul (DiffeRet<>), (Call<(SameTypesFunc<"cosh">), [ReadNone,NoUnwind]> $x))],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x),
                  ["sinhf"],
                  [(FMul (DiffeRet<>), (Call<(SameTypesFunc<"coshf">), [ReadNone,NoUnwind]> $x))],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["exp10"],
                  [(FMul (FMul (DiffeRet<>), (Call<(SameFunc), [ReadNone,NoUnwind]> $x) ), (ConstantFP<"2.30258509299404568401799145468"> $x))],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x),
                  ["tan", "tanf", "tanl"],
                  [(FMul (DiffeRet<>), (FAdd (ConstantFP<"1.0"> $x), (FMul(Call<(SameFunc), [ReadNone,NoUnwind]> $x):$c, $c)))],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x, $y),
                  ["remainder"],
                  [
                    (DiffeRet<>),
                    (FMul (FNeg (DiffeRet<>)), (Intrinsic<"round"> (FDiv $x, $y)))
                  ],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x),
                  ["asin", "asinf", "asinl"],
                  [(FDiv (DiffeRet<>), (Intrinsic<"sqrt"> (FSub (ConstantFP<"1.0"> $x), (FMul $x, $x)))  )] ,
                  (ForwardFromSummedReverse<>)                 
                  >;
def : CallPattern<(Op $x),
                  ["acos", "acosf", "acosl"],
                  [(FNeg (FDiv (DiffeRet<>), (Intrinsic<"sqrt"> (FSub (ConstantFP<"1.0"> $x), (FMul $x, $x)))  ))],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x),
                  ["log1p", "log1pf", "log1pl"],
                  [(FDiv (DiffeRet<>), (FAdd $x, (ConstantFP<"1.0"> $x)))],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x),
                  ["expm1"],
                  [(FMul (Intrinsic<"exp"> $x), (DiffeRet<>))],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x), ["jl_rem2pi", "jl_rem2pif", "jl_rem2pil"],[(DiffeRet<>)],
                  (ForwardFromSummedReverse<>)>;

// Unnormalized sinc(x) = sin(x)/x
def : CallPattern<(Op $x),
                  ["sinc", "sincf", "sincl"],
                  [(Select (FCmpOEQ $x, (ConstantFP<"0"> $x)),
                        (ConstantFP<"0"> $x),
                  (FMul (DiffeRet<>), (FDiv (FSub (Intrinsic<"cos"> $x), (Call<(SameFunc), [ReadNone,NoUnwind]> $x)), $x)))],
                  (ForwardFromSummedReverse<>)>;

// Normalized sinc(x) = sin(pi x)/(pi x)
def : CallPattern<(Op $x),
                  ["sincn", "sincnf", "sincnl"],
                  [
                    (Select (FCmpOEQ $x, (ConstantFP<"0"> $x)),
                        (ConstantFP<"0"> $x),
                  (FMul (DiffeRet<>), (FDiv (FSub (Intrinsic<"cos"> (FMul (ConstantFP<"3.1415926535897962684626433"> $x), $x)), (Call<(SameFunc), [ReadNone,NoUnwind]> $x)), $x)))],
                  (ForwardFromSummedReverse<>)>;

// Normalized Faddeeva_erfcx_re(x) = Exp[z^2] Erfc[z] -> 2 dx ( x f(x) - 1/sqrt(pi))
def : CallPattern<(Op $x),
                  ["Faddeeva_erfcx_re"],
                  [
                  (FMul (DiffeRet<>), (FMul (ConstantFP<"2.0"> $x), (FSub (FMul $x, (Call<(SameFunc), [ReadNone,NoUnwind]> $x)), (ConstantFP<"0.56418958354775628694807945156077258584405062932900"> $x) )))
                  ],
                  (ForwardFromSummedReverse<>)>;

def : CallPattern<(Op $x, $y),
                  ["fmod", "fmodf", "fmodl"],
                  [
                  (DiffeRet<>),
                  (FMul (DiffeRet<>), (FNeg (Intrinsic<"copysign"> (Intrinsic<"floor"> (Intrinsic<"fabs"> (FDiv $x, $y))), (FDiv $x, $y))))
                  ],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["__fd_sincos_1", "__fd_sincos_1f", "__fd_sincos_1l"],
                  [
                  (
                   ArrayRet (FMul (ExtractValue<[1]> (Call<(SameFunc), [ReadNone,NoUnwind]> $x):$callset), (DiffeRet<>) ),
                            (FNeg (FMul (ExtractValue<[0]> $callset), (DiffeRet<>) ) )
                  )
                  ],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["sincospi", "sincospif", "sincospil"],
                  [
                  (
                   ArrayRet (FMul (ConstantFP<"3.1415926535897962684626433"> $x), (FMul (ExtractValue<[1]> (Call<(SameFunc), [ReadNone,NoUnwind]> $x):$callset), (DiffeRet<>) ) ),
                            (FMul (ConstantFP<"3.1415926535897962684626433"> $x),  (FNeg (FMul (ExtractValue<[0]> $callset), (DiffeRet<>) ) ) )
                  )
                  ],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["sinpi", "sinpif", "sinpil", "cospi", "cospif", "cospil"],
                  [
                  (
                   FMul (ConstantFP<"3.1415926535897962684626433"> $x), (FMul (Call<(SameFunc), [ReadNone,NoUnwind]> (FAdd $x, (ConstantFP<"0.5"> $x))), (DiffeRet<>) )
                  )
                  ],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["j0"],
                  [(FMul (DiffeRet<>), (FNeg (Call<(SameTypesFunc<"j1">), [ReadNone,NoUnwind]> $x) ))],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x),
                  ["j0f"],
                  [(FMul (DiffeRet<>), (FNeg (Call<(SameTypesFunc<"j1f">), [ReadNone,NoUnwind]> $x) ))],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x),
                  ["y0"],
                  [(FMul (DiffeRet<>), (FNeg (Call<(SameTypesFunc<"y1">), [ReadNone,NoUnwind]> $x) ))],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x),
                  ["y0f"],
                  [(FMul (DiffeRet<>), (FNeg (Call<(SameTypesFunc<"y1f">), [ReadNone,NoUnwind]> $x) ))],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["j1"],
                  [(FMul (DiffeRet<>), (FMul (ConstantFP<"0.5"> $x), (FSub (Call<(SameTypesFunc<"j0">), [ReadNone,NoUnwind]> $x), (Call<(PrependArgTypesFunc<"jn", ["Int32"]>), [ReadNone,NoUnwind]> (ConstantInt<2, 32>), $x) )))],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x),
                  ["j1f"],
                  [(FMul (DiffeRet<>), (FMul (ConstantFP<"0.5"> $x), (FSub (Call<(SameTypesFunc<"j0f">), [ReadNone,NoUnwind]> $x), (Call<(PrependArgTypesFunc<"jnf", ["Int32"]>), [ReadNone,NoUnwind]> (ConstantInt<2, 32>), $x) )))],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["y1"],
                  [(FMul (DiffeRet<>), (FMul (ConstantFP<"0.5"> $x), (FSub (Call<(SameTypesFunc<"y0">), [ReadNone,NoUnwind]> $x), (Call<(PrependArgTypesFunc<"yn", ["Int32"]>), [ReadNone,NoUnwind]> (ConstantInt<2, 32>), $x) )))],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x),
                  ["y1f"],
                  [(FMul (DiffeRet<>), (FMul (ConstantFP<"0.5"> $x), (FSub (Call<(SameTypesFunc<"y0f">), [ReadNone,NoUnwind]> $x), (Call<(PrependArgTypesFunc<"ynf", ["Int32"]>), [ReadNone,NoUnwind]> (ConstantInt<2, 32>), $x) )))],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $n, $x),
                  ["jn","jnf","yn","ynf"],
                  [
                    (InactiveArg),
                    (FMul (DiffeRet<>), (FMul (ConstantFP<"0.5"> $x), (FSub (Call<(SameFunc), [ReadNone,NoUnwind]> (ISub $n, (ConstantInt<1> $n)), $x), (Call<(SameFunc), [ReadNone,NoUnwind]> (IAdd $n, (ConstantInt<1> $n)), $x))))
                  ],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["erf"],
                  [
                    (FMul (DiffeRet<>), (FMul (ConstantFP<"1.1283791670955125738961589031215451716881012586580"> $x), (Intrinsic<"exp"> (FNeg (FMul $x, $x)))))
                  ],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x),
                  ["erfi"],
                  [
                    (FMul (DiffeRet<>), (FMul (ConstantFP<"1.1283791670955125738961589031215451716881012586580"> $x), (Intrinsic<"exp"> (FMul $x, $x))))
                  ],
                  (ForwardFromSummedReverse<>)
                  >;
def : CallPattern<(Op $x),
                  ["erfc"],
                  [
                    (FMul (DiffeRet<>), (FMul (ConstantFP<"-1.1283791670955125738961589031215451716881012586580"> $x), (Intrinsic<"exp"> (FNeg (FMul $x, $x)))))
                  ],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["Faddeeva_erf"],
                  [
                    (CFMul (DiffeRet<>), (CFMul (ConstantCFP<"1.1283791670955125738961589031215451716881012586580","0"> $x), (CFExp (CFNeg (CFMul $x, $x)))))
                  ],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["Faddeeva_erfi"],
                  [
                    (CFMul (DiffeRet<>), (CFMul (ConstantCFP<"1.1283791670955125738961589031215451716881012586580","0"> $x), (CFExp (CFMul $x, $x))))
                  ],
                  (ForwardFromSummedReverse<>)
                  >;

def : CallPattern<(Op $x),
                  ["Faddeeva_erfc"],
                  [
                    (CFMul (DiffeRet<>), (CFMul (ConstantCFP<"-1.1283791670955125738961589031215451716881012586580","0"> $x), (CFExp (CFNeg (CFMul $x, $x)))))
                  ],
                  (ForwardFromSummedReverse<>)
                  >;

def : IntrPattern<(Op $x),
                  [["sin"]],
                  [(FMul (DiffeRet<>), (Intrinsic<"cos"> $x))]  ,
                  (ForwardFromSummedReverse<>)                
                  >;

def : IntrPattern<(Op $x),
                  [["cos"]],
                  [(FMul (DiffeRet<>), (FNeg (Intrinsic<"sin"> $x)))]       ,
                  (ForwardFromSummedReverse<>)           
                  >;

def : IntrPattern<(Op $x),
                  [["exp"]],
                  [(FMul (DiffeRet<>), (Call<(SameFunc)> $x))],
                  (ForwardFromSummedReverse<>)                  
                  >;

def : IntrPattern<(Op $x),
                  [["exp2"], ["nvvm_ex2_approx_ftz_f"], ["nvvm_ex2_approx_f"], ["nvvm_ex2_approx_d"]],
                  [(FMul (DiffeRet<>), (FMul (ConstantFP<"0.6931471805599453"> $x), (Call<(SameFunc)> $x)))]                  ,
                  (ForwardFromSummedReverse<>)
                  >;

def : IntrPattern<(Op $x),
                  [["log"]],
                  [(FDiv (DiffeRet<>), $x)],
                  (ForwardFromSummedReverse<>)
                  >;

def : IntrPattern<(Op $x),
                  [["log2"]],
                  [(FDiv (DiffeRet<>), (FMul (ConstantFP<"0.6931471805599453"> $x), $x))],
                  (ForwardFromSummedReverse<>)
                  >;

def : IntrPattern<(Op $x),
                  [["log10"]],
                  [(FDiv (DiffeRet<>), (FMul (ConstantFP<"2.302585092994046"> $x), $x))],
                  (ForwardFromSummedReverse<>)
                  >;

def : IntrPattern<(Op $x, $y),
                  [["copysign"]],
                  [
                    (FMul (DiffeRet<>), (FMul (Call<(SameFunc)> (ConstantFP<"1.0"> $x), $x), (Call<(SameFunc)> (ConstantFP<"1.0"> $y), $y))),
                    (InactiveArg)
                  ],
                  (ForwardFromSummedReverse<>)
                  >;

def : IntrPattern<(Op $x, $y),
                  [["minnum"], ["nvvm_fmin_f"], ["nvvm_fmin_d"], ["nvvm_fmin_ftz_f"], ["x86_sse_min_ss", "", "9"], ["x86_sse_min_ps", "", "9"], ["minimum", "15", ""]],
                  [
                    (Select (FCmpOLT $x, $y), (DiffeRet<>), (ConstantFP<"0"> $x)),
                    (Select (FCmpOLT $x, $y), (ConstantFP<"0"> $x), (DiffeRet<>))
                  ],
                  (Select (FCmpOLT $x, $y), (ShadowOrZero $x), (ShadowOrZero $y))
                  >;

def : IntrPattern<(Op $x, $y),
                  [["maxnum"], ["nvvm_fmax_f"], ["nvvm_fmax_d"], ["nvvm_fmax_ftz_f"], ["x86_sse_max_ss", "", "9"], ["x86_sse_max_ps", "", "9"], ["maximum", "15", ""]],
                  [
                    (Select (FCmpOLT $x, $y), (ConstantFP<"0"> $x), (DiffeRet<>)),
                    (Select (FCmpOLT $x, $y), (DiffeRet<>), (ConstantFP<"0"> $x))
                  ],
                  (Select (FCmpOLT $x, $y), (ShadowOrZero $y), (ShadowOrZero $x))
                  >;

def : IntrPattern<(Op $x),
                  [["fabs"]],
                  [
                    (FMul (DiffeRet<>), (Select (FCmpOLT $x, (ConstantFP<"0"> $x)), (ConstantFP<"-1"> $x), (ConstantFP<"1"> $x)))
                  ],
                  (ForwardFromSummedReverse<>)
                  >;

def : IntrPattern<(Op $x),
                  [["sqrt"]],
                  [(Select (FCmpUEQ $x, (ConstantFP<"0"> $x)), (ConstantFP<"0"> $x), (FDiv (DiffeRet<>), (FMul (ConstantFP<"2"> $x), (Call<(SameFunc), [ReadNone,NoUnwind]> $x))))],
                  (ForwardFromSummedReverse<>)
                 >;

def : IntrPattern<(Op $x, $y),
                  [["pow"]],
                  [
                    (CheckedMul (DiffeRet<>), (FMul $y, (Call<(SameFunc)> $x, (FSub $y, (ConstantFP<"1"> $y))))),
                    (CheckedMul (DiffeRet<>), (FMul (Call<(SameFunc)> $x, $y), (Intrinsic<"log"> $x)
                    ))
                  ],
                  (ForwardFromSummedReverse<>)
                 >;

def : IntrPattern<(Op $x, $y),
                  [["powi"]],
                  [
                    (Select 
                      (Or (ICmpEQ $y, (ConstantInt<0> $y) ), (And (GlobalExpr<"ConstantInt::get(Type::getInt1Ty(gutils->newFunc->getContext()), EnzymeStrongZero)">), (FCmpOEQ (DiffeRet<>), (ConstantFP<"0"> $x) ) ) ),
                      (ConstantFP<"0"> $x),
                      (FMul (DiffeRet<>),
                        (FMul (SIToFP $y, (TypeOf<> $x)), (Call<(SameFunc)> $x, (ISub $y, (ConstantInt<1> $y))))
                      )
                     ),
                    (InactiveArg)
                  ],
                  (ForwardFromSummedReverse<>)
                 >;

def : IntrPattern<(Op $x, $y, $z),
                  [["fma"], ["fmuladd"]],
                  [
                    (FMul (DiffeRet<>), $y),
                    (FMul (DiffeRet<>), $x),
                    (DiffeRet<>)
                  ],
                  (ForwardFromSummedReverse<>)
                 >;

def : IntrPattern<(Op $x, $y),
                  [["experimental_vector_reduce_v2_fadd", "", "11"], ["vector_reduce_fadd", "12", ""]],
                  [
                    (DiffeRet<>),
                    (VectorSplat (VectorSize<> (TypeOf<> $y)), (DiffeRet<>))
                  ],
                  (Call<(SameFunc)> (ShadowOrZero $x), (ShadowOrZero $y))
                 >;

include "BlasDerivatives.td"                  
