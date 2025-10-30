class Type;
class ForwardFromSummedReverseInternal<int unused_> {
  int unused = unused_;  
}
def ForwardFromSummedReverse : ForwardFromSummedReverseInternal<0>;

class Operation<bit usesPrimal_, bit usesShadow_, bit usesCustom_=0> {
  bit usesPrimal = usesPrimal_;
  bit usesShadow = usesShadow_;
  bit usesCustom = usesCustom_;
}
class ConstantFP<string val> : Operation</*primal*/0, /*shadow*/0> {
  string value = val;
}

class StaticSelect<string condition_> : Operation</*primal*/0, /*shadow*/0, /*custom*/0> {
  string condition = condition_;
}

def SelectIfActive : StaticSelect<"!gutils->isConstantValue(imVal)">;

class Attribute<string name_> {
  string name = name_;
}
def ReadNone : Attribute<"ReadNone">;
def ReadOnly : Attribute<"ReadOnly">;
def WriteOnly : Attribute<"WriteOnly">;
def NoUnwind : Attribute<"NoUnwind">;

class CallPattern<dag patternToMatch, list<string> funcNames, list<dag> resultOps, dag forwardOps, list<Attribute> attrs=[]> {
  dag PatternToMatch = patternToMatch;
  list<string> names = funcNames;
  list<dag> ArgDerivatives = resultOps;
  dag ArgDuals = forwardOps;
  list<Attribute> FnAttrs = attrs;
}

class IntrPattern<dag patternToMatch, list<list<string>> funcNames, list<dag> resultOps, dag forwardOps> {
  dag PatternToMatch = patternToMatch;
  list<list<string>> names = funcNames;
  list<dag> ArgDerivatives = resultOps;
  dag ArgDuals = forwardOps;
}


class BinopPattern<dag patternToMatch, string funcName, int minVer_, int maxVer_, list<dag> resultOps, dag forwardOps> {
  dag PatternToMatch = patternToMatch;
  string name = funcName;
  int minVer = minVer_;
  int maxVer = maxVer_;
  list<dag> ArgDerivatives = resultOps;
  dag ArgDuals = forwardOps;
}

class InstPattern<dag patternToMatch, string funcName, int minVer_, int maxVer_, list<dag> resultOps, dag forwardOps> {
  dag PatternToMatch = patternToMatch;
  string name = funcName;
  int minVer = minVer_;
  int maxVer = maxVer_;
  list<dag> ArgDerivatives = resultOps;
  dag ArgDuals = forwardOps;
}

class Inst<string mnemonic>  : Operation</*primal*/1, /*shadow*/0> {
  string name = mnemonic;
}

def TypeOf : Operation</*primal*/0, /*shadow*/0> {
}
def VectorSize : Operation</*primal*/0, /*shadow*/0> {
}

// Define ops to rewrite.

def CheckedMul : Inst<"CheckedMul">;
def CheckedDiv : Inst<"CheckedDiv">;
def InsertElement : Inst<"InsertElement">;
def ShuffleVector : Inst<"ShuffleVector">;
def VectorSplat : Inst<"VectorSplat">;
def SIToFP : Inst<"SIToFP">;
def BitCast : Inst<"BitCast">;
def Or : Inst<"Or">;
def And : Inst<"And">;
def IAdd : Inst<"Add">;
def ISub : Inst<"Sub">;
def FDiv : Inst<"FDiv">;
def FAdd : Inst<"FAdd">;
def FSub : Inst<"FSub">;
def FMul : Inst<"FMul">;
def FNeg : Inst<"FNeg">;
def Freeze : Inst<"Freeze">;
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

class SubRoutine<dag patternToMatch, dag insts_> : Operation</*primal*/1,/*shadow*/0> {
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

class MultiReturn : Operation</*primal*/1,/*shadow*/0> {
  bit struct;
  bit useRetType;
}
def RetMultiReturnRet : MultiReturn {
  bit struct = 0;
  bit useRetType = 1;
}
def StructRet : MultiReturn {
  bit struct = 1;
  bit useRetType = 0;
}
def ArrayRet : MultiReturn {
  bit struct = 0;
  bit useRetType = 0;
}

def CFAdd : SubRoutine<(Op (Op $re1, $im1):$z1, (Op $re2, $im2):$z2),
                  (ArrayRet
                    (FAdd $re1, $re2),
                    (FAdd $im1, $im2)
                  )>;
def CFSub : SubRoutine<(Op (Op $re1, $im1):$z1, (Op $re2, $im2):$z2),
                  (ArrayRet
                    (FSub $re1, $re2),
                    (FSub $im1, $im2)
                  )>;

def CFMul_splat : SubRoutine<(Op $re1, $im1, $re2, $im2),
                  (ArrayRet
                    (FSub (FMul $re1, $re2), (FMul $im1, $im2)),
                    (FAdd (FMul $re1, $im2), (FMul $re2, $im1))
                  )>;

def CFMul : SubRoutine<(Op (Op $re1, $im1):$z1, (Op $re2, $im2):$z2),
                  (CFMul_splat $re1, $im1, $re2, $im2)>;


def CFDiv_splat : SubRoutine<(Op $re1, $im1, $re2, $im2),
                  (ArrayRet
                    (FDiv (FAdd (FMul $re1, $re2), (FMul $im1, $im2)), (FAdd (FMul $re2, $re2), (FMul $im2, $im2)):$denom),
                    (FDiv (FSub (FMul $re2, $im1), (FMul $re1, $im2)), $denom)
                  )>;

def CFDiv : SubRoutine<(Op (Op $re1, $im1):$z1, (Op $re2, $im2):$z2),
                  (CFDiv_splat $re1, $im1, $re2, $im2)>;

def CFNeg : SubRoutine<(Op (Op $re, $im):$z),
    (ArrayRet
      (FNeg $re),
      (FNeg $im)
    )>;

def Complex : SubRoutine<(Op $x),
    (ArrayRet
      $x,
      (ConstantFP<"0"> $x)
    )>;

def Conj : SubRoutine<(Op (Op $re, $im):$z),
    (ArrayRet
      $re,
      (FNeg $im)
    )>;

def CFExp : SubRoutine<(Op (Op $re, $im):$z),
    (ArrayRet
      (FMul (FExp $re):$exp, (FCos $im)),
      (FMul $exp, (FSin $im))
    )>;


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

// Set return to arg[0]
// Same argument types
class ArgAsRetTypesFunc<string name_> {
  string name = name_;
}

// Specify that a given argument is inactive, aka not differentiable
// By default this argument tells Enzyme that it must always be inactive
// from the function semantics.
// One can alternatively pass the asserting=1 flag, causing an assertion
// error to be thrown if the value is actually active.
class InactiveArgSpec {
  bit asserting;
}
def InactiveArg : InactiveArgSpec {
  bit asserting = 0;
}
def AssertingInactiveArg : InactiveArgSpec {
  bit asserting = 1;
}

class GlobalExpr<bit uses_primal, bit uses_shadow, string val> : Operation<uses_primal, uses_shadow>{
  string value = val;
}
// Create a bit-mask whose bitwise "and" with a floating-point number gives
// the largest power of two not larger than that number (in terms of absolute values).
def MantissaMaskOfReturnForFrexp : GlobalExpr</*primal*/0, /*shadow*/0, [{
  auto ty = call.getType();
  unsigned low = 0;
  unsigned high = 0;
  unsigned tsize = 0;
  if (ty->isHalfTy()) {
    tsize = 16;
    high = tsize - 1;
    low = high - 5;
  } else if (ty->isFloatTy()) {
    tsize = 32;
    high = tsize - 1;
    low = high - 8;
  } else if (ty->isDoubleTy()) {
    tsize = 64;
    high = tsize - 1;
    low = high - 11;
  } else if (ty->isX86_FP80Ty()) {
    tsize = 80;
    high = tsize - 1;
    low = high - 16;
    // x86_fp80 has only 15 exponent bits, but we must also
    // retain the most-significant bit of the mantissa as
    // there is no implicit leading
  } else if (ty->isBFloatTy()) {
    tsize = 16;
    high = tsize - 1;
    low = high - 8;
  } else if (ty->isFP128Ty()) {
    tsize = 128;
    high = tsize - 1;
    low = high - 15;
  } else {
    llvm_unreachable("Unknown type within frexp");
  }
  APInt eval = APInt::getBitsSet(tsize, low, high);
  auto ity = IntegerType::get(ty->getContext(), tsize);
  ConstantInt::get(ity, eval);
}]>;

def Zero : Operation</*primal*/0, /*shadow*/0> {
}
class ConstantCFP<string rval, string ival> : Operation</*primal*/0, /*shadow*/0> {
  string rvalue = rval;
  string ivalue = ival;
}
class ConstantInt<int value_, int bitwidth_=0> : Operation</*primal*/0, /*shadow*/0> {
  int value = value_;
  int bitwidth=bitwidth_;
}
class Undef : Operation</*primal*/0, /*shadow*/0> {
}

class DiffeRetIndex<list<int> indices_> {
  list<int> indices = indices_;
}
def DiffeRet : DiffeRetIndex<[-1]>;

def Shadow : Operation</*primal*/0, /*shadow*/1> {
}

//  1 / y -> -1/y^2 dy
def : CallPattern<(Op $y),
                  ["__nv_frcp_rd", "__nv_frcp_rn", "__nv_frcp_ru", "__nv_frcp_rz",
                   "__nv_drcp_rd", "__nv_drcp_rn", "__nv_drcp_ru", "__nv_drcp_rz"],
                    [
                      (FNeg (CheckedDiv (DiffeRet), (FMul $y, $y)))
                    ],
                    (ForwardFromSummedReverse),
                    [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["atan", "atanf", "atanl", "__fd_atan_1"],
                  [(CheckedDiv (DiffeRet), (FAdd (FMul $x, $x), (ConstantFP<"1.0"> $x)))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $y, $x),
                  ["atan2", "atan2f", "atan2l", "__nv_atan2", "__nv_atan2f", "__nv_atan2l"],
                  [
                  (FDiv (FMul (DiffeRet), $x), (FAdd (FMul $x, $x), (FMul $y, $y))),
                  (FNeg (FDiv (FMul (DiffeRet), $y), (FAdd (FMul $x, $x), (FMul $y, $y))))
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["cbrt", "cbrtf", "cbrtl"],
                  [(FDiv (FMul (DiffeRet), (Call<(SameFunc), [ReadNone,NoUnwind]> $x) ), (FMul (ConstantFP<"3.0"> $x), $x))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x, $y),
                  ["hypot", "hypotf", "hypotl", "__hypot_finite"],
                  [
                    (FDiv (FMul (DiffeRet), $x), (Call<(SameFunc), [ReadNone,NoUnwind]> $x, $y)),
                    (FDiv (FMul (DiffeRet), $y), (Call<(SameFunc), [ReadNone,NoUnwind]> $x, $y))
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["tanh"],
                  [(FDiv (DiffeRet), (FMul(Call<(SameTypesFunc<"cosh">), [ReadNone,NoUnwind]> $x):$c, $c))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["tanhf"],
                  [(FDiv (DiffeRet), (FMul(Call<(SameTypesFunc<"coshf">), [ReadNone,NoUnwind]> $x):$c, $c))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["tanhl"],
                  [(FDiv (DiffeRet), (FMul(Call<(SameTypesFunc<"coshl">), [ReadNone,NoUnwind]> $x):$c, $c))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["cosh"],
                  [(FMul (DiffeRet), (Call<(SameTypesFunc<"sinh">), [ReadNone,NoUnwind]> $x))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["coshf"],
                  [(FMul (DiffeRet), (Call<(SameTypesFunc<"sinhf">), [ReadNone,NoUnwind]> $x))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["coshl"],
                  [(FMul (DiffeRet), (Call<(SameTypesFunc<"sinhl">), [ReadNone,NoUnwind]> $x))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["sinh"],
                  [(FMul (DiffeRet), (Call<(SameTypesFunc<"cosh">), [ReadNone,NoUnwind]> $x))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["sinhf"],
                  [(FMul (DiffeRet), (Call<(SameTypesFunc<"coshf">), [ReadNone,NoUnwind]> $x))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["sinhl"],
                  [(FMul (DiffeRet), (Call<(SameTypesFunc<"coshl">), [ReadNone,NoUnwind]> $x))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["asinh", "asinhf", "asinhl", "__nv_asinh", "__nv_asinhf"],
                  [(FDiv (DiffeRet), (Intrinsic<"sqrt"> (FAdd (FMul $x, $x), (ConstantFP<"1.0"> $x)))  )] ,
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["acosh", "acoshf", "acoshl", "__nv_acosh", "__nv_acoshf"],
                  [(FDiv (DiffeRet), (Intrinsic<"sqrt"> (FSub (FMul $x, $x), (ConstantFP<"1.0"> $x)))  )] ,
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["atanh", "atanhf", "atanhl", "__nv_atanh", "__nv_atanhf"],
                  [(FDiv (DiffeRet), (FSub (ConstantFP<"1.0"> $x), (FMul $x, $x))  )] ,
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["__nv_fast_expf"],
                  [(FMul (DiffeRet), (Call<(SameFunc), [ReadNone,NoUnwind]> $x) )],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["exp10"],
                  [(FMul (FMul (DiffeRet), (Call<(SameFunc), [ReadNone,NoUnwind]> $x) ), (ConstantFP<"2.30258509299404568401799145468"> $x))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["tan", "tanf", "tanl"],
                  [(FMul (DiffeRet), (FAdd (ConstantFP<"1.0"> $x), (FMul(Call<(SameFunc), [ReadNone,NoUnwind]> $x):$c, $c)))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x, $y),
                  ["remainder"],
                  [
                    (DiffeRet),
                    (FMul (FNeg (DiffeRet)), (Intrinsic<"round"> (FDiv $x, $y)))
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["asin", "asinf", "asinl", "__nv_asin", "__nv_asinf", "__nv_asinl"],
                  [(FDiv (DiffeRet), (Intrinsic<"sqrt"> (FSub (ConstantFP<"1.0"> $x), (FMul $x, $x)))  )] ,
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]              
                  >;
def : CallPattern<(Op $x),
                  ["acos", "acosf", "acosl", "__nv_acos", "__nv_acosf", "__nv_acosl"],
                  [(FNeg (FDiv (DiffeRet), (Intrinsic<"sqrt"> (FSub (ConstantFP<"1.0"> $x), (FMul $x, $x)))  ))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["log1p", "log1pf", "log1pl"],
                  [(FDiv (DiffeRet), (FAdd $x, (ConstantFP<"1.0"> $x)))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["expm1", "expm1f", "expm1l"],
                  [(FMul (Intrinsic<"exp"> $x), (DiffeRet))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x), ["jl_rem2pi", "jl_rem2pif", "jl_rem2pil"],[(DiffeRet)],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

// Unnormalized sinc(x) = sin(x)/x
def : CallPattern<(Op $x),
                  ["sinc", "sincf", "sincl"],
                  [(Select (FCmpOEQ $x, (ConstantFP<"0"> $x)),
                        (ConstantFP<"0"> $x),
                  (FMul (DiffeRet), (FDiv (FSub (Intrinsic<"cos"> $x), (Call<(SameFunc), [ReadNone,NoUnwind]> $x)), $x)))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

// Normalized sinc(x) = sin(pi x)/(pi x)
def : CallPattern<(Op $x),
                  ["sincn", "sincnf", "sincnl"],
                  [
                    (Select (FCmpOEQ $x, (ConstantFP<"0"> $x)),
                        (ConstantFP<"0"> $x),
                  (FMul (DiffeRet), (FDiv (FSub (Intrinsic<"cos"> (FMul (ConstantFP<"3.1415926535897962684626433"> $x), $x)), (Call<(SameFunc), [ReadNone,NoUnwind]> $x)), $x)))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

// Normalized Faddeeva_erfcx_re(x) = Exp[z^2] Erfc[z] -> 2 dx ( x f(x) - 1/sqrt(pi))
def : CallPattern<(Op $x),
                  ["Faddeeva_erfcx_re"],
                  [
                  (FMul (DiffeRet), (FMul (ConstantFP<"2.0"> $x), (FSub (FMul $x, (Call<(SameFunc), [ReadNone,NoUnwind]> $x)), (ConstantFP<"0.56418958354775628694807945156077258584405062932900"> $x) )))
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x, $y),
                  ["fmod", "fmodf", "fmodl", "__nv_fmod", "__nv_fmodf", "__nv_fmodl"],
                  [
                  (DiffeRet),
                  (CheckedMul (DiffeRet), (FNeg (Intrinsic<"copysign"> (Intrinsic<"floor"> (Intrinsic<"fabs"> (FDiv $x, $y):$div)), $div)))
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x, $integral_part_ptr),
                  ["modf", "modff", "modfl"],
                  [
                    (DiffeRet),
                    (InactiveArg)
                  ],
                  (ForwardFromSummedReverse),
                  [ReadOnly, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["__fd_sincos_1", "__fd_sincos_1f", "__fd_sincos_1l"],
                  [
                  (
                   ArrayRet (FMul (ExtractValue<[1]> (Call<(SameFunc), [ReadNone,NoUnwind]> $x):$callset), (DiffeRet) ),
                            (FNeg (FMul (ExtractValue<[0]> $callset), (DiffeRet) ) )
                  )
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["sincospi", "sincospif", "sincospil"],
                  [
                  (
                   ArrayRet (FMul (ConstantFP<"3.1415926535897962684626433"> $x), (FMul (ExtractValue<[1]> (Call<(SameFunc), [ReadNone,NoUnwind]> $x):$callset), (DiffeRet) ) ),
                            (FMul (ConstantFP<"3.1415926535897962684626433"> $x),  (FNeg (FMul (ExtractValue<[0]> $callset), (DiffeRet) ) ) )
                  )
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x, $y),
                  ["xlogy_jl"],
                  [
                    (FMul (Intrinsic<"log"> $y), (DiffeRet)),
                    (Select (FCmpOEQ $x, (ConstantFP<"0"> $x)),
                        (ConstantFP<"0"> $x),
                        (FMul (FDiv $x, $y), (DiffeRet))
                    )
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["tgamma"],
                  [
                  (
                   FMul (Call<(ArgAsRetTypesFunc<"digamma">), [ReadNone,NoUnwind]> $x), (DiffeRet)
                  )
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["logabsgamma", "lgamma"],
                  [
                  (
                   ArrayRet (FMul (Call<(ArgAsRetTypesFunc<"digamma">), [ReadNone,NoUnwind]> $x), (DiffeRet) ),
                            (InactiveArg)
                  )
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["logabsgammaf", "lgammaf"],
                  [
                  (
                   ArrayRet (FMul (Call<(ArgAsRetTypesFunc<"digammaf">), [ReadNone,NoUnwind]> $x), (DiffeRet) ),
                            (InactiveArg)
                  )
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["sinpi", "sinpif", "sinpil", "cospi", "cospif", "cospil"],
                  [
                  (
                   FMul (ConstantFP<"3.1415926535897962684626433"> $x), (FMul (Call<(SameFunc), [ReadNone,NoUnwind]> (FAdd $x, (ConstantFP<"0.5"> $x))), (DiffeRet) )
                  )
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["j0"],
                  [(FMul (DiffeRet), (FNeg (Call<(SameTypesFunc<"j1">), [ReadNone,NoUnwind]> $x) ))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["j0f"],
                  [(FMul (DiffeRet), (FNeg (Call<(SameTypesFunc<"j1f">), [ReadNone,NoUnwind]> $x) ))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["y0"],
                  [(FMul (DiffeRet), (FNeg (Call<(SameTypesFunc<"y1">), [ReadNone,NoUnwind]> $x) ))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["y0f"],
                  [(FMul (DiffeRet), (FNeg (Call<(SameTypesFunc<"y1f">), [ReadNone,NoUnwind]> $x) ))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["j1"],
                  [(FMul (DiffeRet), (FMul (ConstantFP<"0.5"> $x), (FSub (Call<(SameTypesFunc<"j0">), [ReadNone,NoUnwind]> $x), (Call<(PrependArgTypesFunc<"jn", ["Int32"]>), [ReadNone,NoUnwind]> (ConstantInt<2, 32>), $x) )))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["j1f"],
                  [(FMul (DiffeRet), (FMul (ConstantFP<"0.5"> $x), (FSub (Call<(SameTypesFunc<"j0f">), [ReadNone,NoUnwind]> $x), (Call<(PrependArgTypesFunc<"jnf", ["Int32"]>), [ReadNone,NoUnwind]> (ConstantInt<2, 32>), $x) )))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["y1"],
                  [(FMul (DiffeRet), (FMul (ConstantFP<"0.5"> $x), (FSub (Call<(SameTypesFunc<"y0">), [ReadNone,NoUnwind]> $x), (Call<(PrependArgTypesFunc<"yn", ["Int32"]>), [ReadNone,NoUnwind]> (ConstantInt<2, 32>), $x) )))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["y1f"],
                  [(FMul (DiffeRet), (FMul (ConstantFP<"0.5"> $x), (FSub (Call<(SameTypesFunc<"y0f">), [ReadNone,NoUnwind]> $x), (Call<(PrependArgTypesFunc<"ynf", ["Int32"]>), [ReadNone,NoUnwind]> (ConstantInt<2, 32>), $x) )))],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $n, $x),
                  ["jn","jnf","yn","ynf"],
                  [
                    (InactiveArg),
                    (FMul (DiffeRet), (FMul (ConstantFP<"0.5"> $x), (FSub (Call<(SameFunc), [ReadNone,NoUnwind]> (ISub $n, (ConstantInt<1> $n)), $x), (Call<(SameFunc), [ReadNone,NoUnwind]> (IAdd $n, (ConstantInt<1> $n)), $x))))
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $n, $z),
                  ["cmplx_jn","cmplx_yn"],
                  [
                    (AssertingInactiveArg),
					// Reverse mode needs to return the conjugate
                    (CFMul (DiffeRet), (Conj (CFMul (ConstantCFP<"0.5", "0"> $z), (CFSub (Call<(SameFunc), [ReadNone,NoUnwind]> (FSub $n, (ConstantFP<"1"> $n)), $z), (Call<(SameFunc), [ReadNone,NoUnwind]> (FAdd $n, (ConstantFP<"1"> $n)), $z)))))
                  ],
                  (CFMul (Shadow $z), (CFMul (ConstantCFP<"0.5", "0"> $z), (CFSub (Call<(SameFunc), [ReadNone,NoUnwind]> (FSub $n, (ConstantFP<"1"> $n)), $z), (Call<(SameFunc), [ReadNone,NoUnwind]> (FAdd $n, (ConstantFP<"1"> $n)), $z)))),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["erf","erff","erfl"],
                  [
                    (FMul (DiffeRet), (FMul (ConstantFP<"1.1283791670955125738961589031215451716881012586580"> $x), (Intrinsic<"exp"> (FNeg (FMul $x, $x)))))
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["erfi"],
                  [
                    (FMul (DiffeRet), (FMul (ConstantFP<"1.1283791670955125738961589031215451716881012586580"> $x), (Intrinsic<"exp"> (FMul $x, $x))))
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;
def : CallPattern<(Op $x),
                  ["erfc","erfcf","erfcl"],
                  [
                    (FMul (DiffeRet), (FMul (ConstantFP<"-1.1283791670955125738961589031215451716881012586580"> $x), (Intrinsic<"exp"> (FNeg (FMul $x, $x)))))
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x),
                  ["erfinv","erfinvf","erfinfvl", "__nv_erfinv","__nv_erfinvf","__nv_erfinfvl", ],
                  [
                    (FMul (FMul (ConstantFP<"0.8862269254527580136490837416705725913987747280611935641069038949264556422955160906874753283692723327"> $x), (FMul (Call<(SameFunc), [ReadNone, NoUnwind]> $x):$ei, $ei)), (DiffeRet))
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def ToStruct2 : SubRoutine<(Op (Op $re, $im):$z),
                    (RetMultiReturnRet $re, $im)
                  >;
def : CallPattern<(Op $x, $tbd),
                  ["Faddeeva_erf"],
                  [
                    (ToStruct2 (Conj (CFMul (Conj (DiffeRet)), (CFMul (ConstantCFP<"1.1283791670955125738961589031215451716881012586580","0"> $x), (CFExp (CFNeg (CFMul $x, $x))))))),
                    (InactiveArg) // relerr
                  ],
                  (ToStruct2 (CFMul (Shadow $x), (CFMul (ConstantCFP<"1.1283791670955125738961589031215451716881012586580","0"> $x), (CFExp (CFNeg (CFMul $x, $x)))))),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x, $tbd),
                  ["Faddeeva_erfi"],
                  [
                    (ToStruct2 (Conj (CFMul (Conj (DiffeRet)), (CFMul (ConstantCFP<"1.1283791670955125738961589031215451716881012586580","0"> $x), (CFExp (CFMul $x, $x)))))),
                    (InactiveArg) // relerr
                  ],
                  (ToStruct2 (CFMul (Shadow $x), (CFMul (ConstantCFP<"1.1283791670955125738961589031215451716881012586580","0"> $x), (CFExp (CFMul $x, $x))))),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x, $tbd),
                  ["Faddeeva_erfc"],
                  [
                    (ToStruct2 (Conj (CFMul (Conj (DiffeRet)), (CFMul (ConstantCFP<"-1.1283791670955125738961589031215451716881012586580","0"> $x), (CFExp (CFNeg (CFMul $x, $x))))))),
                    (InactiveArg) // relerr
                  ],
                  (ToStruct2 (CFMul (Shadow $x), (CFMul (ConstantCFP<"-1.1283791670955125738961589031215451716881012586580","0"> $x), (CFExp (CFNeg (CFMul $x, $x)))))),
                  [ReadNone, NoUnwind]
                  >;

// TODO have forward mode tablegen optimize to never emit the call if both shadoworzero's are zero
def : CallPattern<(Op $xre, $xim, $yre, $yim),
                  ["__mulsc3", "__muldc3", "__multc3", "__mulxc3"],
                  [
                    (StructRet
                      (ExtractValue<[0]> (CFMul (DiffeRetIndex<[]>), (ArrayRet $yre, $yim)):$r0),
                      (ExtractValue<[1]> $r0)
                    ),
                    (StructRet
                      (ExtractValue<[0]> (CFMul (DiffeRetIndex<[]>), (ArrayRet $yim, (FNeg $yre))):$r1),
                      (ExtractValue<[1]> $r1)
                    ),
                    (StructRet
                      (ExtractValue<[0]> (CFMul (DiffeRetIndex<[]>), (ArrayRet $xre, $xim)):$r2),
                      (ExtractValue<[1]> $r2)
                    ),
                    (StructRet
                      (ExtractValue<[0]> (CFMul (DiffeRetIndex<[]>), (ArrayRet $xim, (FNeg $xre))):$r3),
                      (ExtractValue<[1]> $r3)
                    )
// Alternate variant calling the same func (but likely much slower, per calling each one)
// This would simply replace the cfmul call, the arrayret/extract code would also need to be
// applied here
//                    (Call<(SameFunc)> (DiffeRet<[0]>), (DiffeRet<[1]>), $yre, $yim),
//                    (Call<(SameFunc)> (DiffeRet<[1]>), (FNeg (DiffeRet<[1]>)), $yre, $yim),
//                    (Call<(SameFunc)> (DiffeRet<[0]>), (DiffeRet<[1]>), $xre, $xim),
//                    (Call<(SameFunc)> (DiffeRet<[1]>), (FNeg (DiffeRet<[1]>)), $yre, $xim)
                  ],
                  (ToStruct2 (CFAdd (CFMul_splat 
                            (SelectIfActive $xre, (Shadow $xre), (Zero $xre)),
                            (SelectIfActive $xim, (Shadow $xim), (Zero $xim)),
                            $yre, $yim),
                          (CFMul_splat 
                            (SelectIfActive $yre, (Shadow $yre), (Zero $yre)),
                            (SelectIfActive $yim, (Shadow $yim), (Zero $yim)),
                            $xre, $xim))),
// Alternate variant below with the same mul fn
//                  (CFAdd (Call<(SameFunc), [ReadNone, NoUnwind]> (ShadowOrZero $xre), (ShadowOrZero $xim), $yre, $yim), (Call<(SameFunc), [ReadNone, NoUnwind]> (ShadowOrZero $yre), (ShadowOrZero $yim), $xre, $xim)),
                  [ReadNone, NoUnwind]
                  >;


// TODO have forward mode tablegen optimize to never emit the call if both shadoworzero's are zero
def : CallPattern<(Op $xre, $xim, $yre, $yim),
                  ["__divsc3", "__divdc3", "__divtc3", "__divxc3"],
                  [
                    (StructRet
                      (ExtractValue<[0]> (CFDiv (DiffeRetIndex<[]>), (ArrayRet $yre, $yim)):$r0),
                      (ExtractValue<[1]> $r0)
                    ),
                    (StructRet
                      (ExtractValue<[0]> (CFDiv (DiffeRetIndex<[]>), (ArrayRet $yim, (FNeg $yre))):$r1),
                      (ExtractValue<[1]> $r1)
                    ),
                    (StructRet
                      (ExtractValue<[0]> (CFNeg (CFDiv (CFMul (DiffeRetIndex<[]>), (ArrayRet $xre, $xim)), (CFMul_splat $yre, $yim, $yre, $yim))):$r2),
                      (ExtractValue<[1]> $r2)
                    ),
                    (StructRet
                      (ExtractValue<[0]> (CFDiv (CFMul (DiffeRetIndex<[]>), (ArrayRet $xim, (FNeg $xim))), (CFMul_splat $yre, $yim, $yre, $yim)):$r2),
                      (ExtractValue<[1]> $r2)
                    )
                  ],
                  (ToStruct2 (CFAdd  (CFDiv_splat 
                            (SelectIfActive $xre, (Shadow $xre), (Zero $xre)),
                            (SelectIfActive $xim, (Shadow $xim), (Zero $xim)),
                            $yre, $yim),
                          (CFMul_splat
                            (SelectIfActive $yre, (Shadow $yre), (Zero $yre)),
                            (SelectIfActive $yim, (Shadow $yim), (Zero $yim)),
                            (ExtractValue<[0]> (CFDiv (ArrayRet (FNeg $xre), (FNeg $xim)), (CFMul_splat $yre, $yim, $yre, $yim))), (ExtractValue<[1]> (CFMul_splat $yre, $yim, $yre, $yim)) ) )),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x, $n),
                  ["scalbn", "scalbnf", "scalbnl", "scalbnln", "scalbnlnf", "scalbnlnl"],
                  [
                    (Call<(SameFunc)> (DiffeRet), $n),
                    (InactiveArg)
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $num, $exp),
                  ["ldexp", "ldexpf", "ldexpl"],
                  [
                    (Call<(SameFunc)> (DiffeRet), $exp),
                    (InactiveArg)
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op $x, $expout),
                  ["frexp", "frexpf", "frexpl"],
                  [
                    (FDiv
                      (DiffeRet),
                      (FMul
                        (BitCast
                          (And (MantissaMaskOfReturnForFrexp):$mask, (BitCast $x, (TypeOf $mask)) ),
                          (TypeOf $x)
                        ),
                        (ConstantFP<"2"> $x)
                      )
                    ),
                    (InactiveArg)
                  ],
                  (ForwardFromSummedReverse),
                  [WriteOnly, NoUnwind]
                  >;

// Handle both the two-arg and one-arg variants of cabs
def : CallPattern<(Op $re, $im),
                  ["cabs", "cabsf", "cabsl"],
                  [
                    (FMul (DiffeRet), (FDiv $re, (Call<(SameFunc), [ReadNone,NoUnwind]> $re, $im))),
                    (FMul (DiffeRet), (FDiv $im, (Call<(SameFunc), [ReadNone,NoUnwind]> $re, $im))),
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;


// This is currently in a helper to use the internal struct-definition form of arrayret,
// rather than the `assuming this function returns a struct` convention if used directly
// within the call pattern.
def Cabs_helper : SubRoutine<(Op (Op $re, $im):$z, $dr, $res),
                    (ArrayRet
                      (FMul $dr, (FDiv $re, $res)),
                      (FMul $dr, (FDiv $im, $res))
                  )>;

def : CallPattern<(Op (Op $x, $y):$z),
                  ["cabs", "cabsf", "cabsl"],
                  [
                    (Cabs_helper $z, (DiffeRet), (Call<(SameFunc), [ReadNone,NoUnwind]> $z))
                  ],
                  (FDiv 
                    (FAdd 
                      (SelectIfActive $x, (FMul (Shadow $x), $x), (Zero $x)),
                      (SelectIfActive $y, (FMul (Shadow $y), $y), (Zero $y))
                    ),
                    (Call<(SameFunc), [ReadNone,NoUnwind]> $z)
                  ),
                  [ReadNone, NoUnwind]
                  >;

def : CallPattern<(Op (Op $x, $y):$z),
                  ["cmplx_inv"],
                  [
					// Reverse mode needs to return the conjugate
                    (Conj (CFDiv (CFNeg (Conj (DiffeRet))), (CFMul $z, $z))),
                  ],
                  (CFDiv (CFNeg (Shadow $z)), (CFMul $z, $z)),
                  [ReadNone, NoUnwind]
                  >;

def : IntrPattern<(Op $x),
                  [["tanh", "19", ""]],
                  [(FDiv (DiffeRet), (FMul(Intrinsic<"cosh"> $x):$c, $c))],
                  (ForwardFromSummedReverse)
                  >;

def : IntrPattern<(Op $x),
                  [["sinh", "19", ""]],
                  [(FMul (DiffeRet), (Intrinsic<"cosh"> $x))],
                  (ForwardFromSummedReverse)
                  >;

def : IntrPattern<(Op $x),
                  [["cosh", "19", ""]],
                  [(FMul (DiffeRet), (Intrinsic<"sinh"> $x))],
                  (ForwardFromSummedReverse)
                  >;

def : IntrPattern<(Op $x),
                  [["sin"]],
                  [(FMul (DiffeRet), (Intrinsic<"cos"> $x))]  ,
                  (ForwardFromSummedReverse)                
                  >;

def : IntrPattern<(Op $x),
                  [["cos"]],
                  [(FMul (DiffeRet), (FNeg (Intrinsic<"sin"> $x)))]       ,
                  (ForwardFromSummedReverse)           
                  >;

def : IntrPattern<(Op $x),
                  [["exp"]],
                  [(FMul (DiffeRet), (Call<(SameFunc)> $x))],
                  (ForwardFromSummedReverse)                  
                  >;

def : IntrPattern<(Op $x),
                  [["exp2"], ["nvvm_ex2_approx_ftz_f"], ["nvvm_ex2_approx_f"], ["nvvm_ex2_approx_d"]],
                  [(FMul (DiffeRet), (FMul (ConstantFP<"0.6931471805599453"> $x), (Call<(SameFunc)> $x)))]                  ,
                  (ForwardFromSummedReverse)
                  >;

def : IntrPattern<(Op $x),
                  [["log"]],
                  [(FDiv (DiffeRet), $x)],
                  (ForwardFromSummedReverse)
                  >;

def : IntrPattern<(Op $x),
                  [["log2"]],
                  [(FDiv (DiffeRet), (FMul (ConstantFP<"0.6931471805599453"> $x), $x))],
                  (ForwardFromSummedReverse)
                  >;

def : IntrPattern<(Op $x),
                  [["log10"]],
                  [(FDiv (DiffeRet), (FMul (ConstantFP<"2.302585092994046"> $x), $x))],
                  (ForwardFromSummedReverse)
                  >;

def : IntrPattern<(Op $x, $y),
                  [["copysign"]],
                  [
                    (FMul (DiffeRet), (FMul (Call<(SameFunc)> (ConstantFP<"1.0"> $x), $x), (Call<(SameFunc)> (ConstantFP<"1.0"> $y), $y))),
                    (InactiveArg)
                  ],
                  (ForwardFromSummedReverse)
                  >;

def : IntrPattern<(Op $x, $y),
                  [["minnum"], ["nvvm_fmin_f"], ["nvvm_fmin_d"], ["nvvm_fmin_ftz_f"], ["x86_sse_min_ss", "", "9"], ["x86_sse_min_ps", "", "9"], ["x86_sse2_min_pd", "", ""], ["minimum", "11", ""]],
                  [
                    (Select (FCmpOLT $x, $y), (DiffeRet), (ConstantFP<"0"> $x)),
                    (Select (FCmpOLT $x, $y), (ConstantFP<"0"> $x), (DiffeRet))
                  ],
                  (Select (FCmpOLT $x, $y), (SelectIfActive $x, (Shadow $x), (Zero $x)), (SelectIfActive $y, (Shadow $y), (Zero $y)))
                  >;

def : IntrPattern<(Op $x, $y),
                  [["maxnum"], ["nvvm_fmax_f"], ["nvvm_fmax_d"], ["nvvm_fmax_ftz_f"], ["x86_sse_max_ss", "", "9"], ["x86_sse_max_ps", "", "9"], ["x86_sse2_max_pd", "", ""], ["maximum", "11", ""]],
                  [
                    (Select (FCmpOLT $x, $y), (ConstantFP<"0"> $x), (DiffeRet)),
                    (Select (FCmpOLT $x, $y), (DiffeRet), (ConstantFP<"0"> $x))
                  ],
                  (Select (FCmpOLT $x, $y), (SelectIfActive $y, (Shadow $y), (Zero $y)), (SelectIfActive $x, (Shadow $x), (Zero $x)))
                  >;

def : CallPattern<(Op $x, $y),
                  ["fdim", "fdimf", "fdiml"],
                  [
                    (Select (FCmpOLT $x, $y), (ConstantFP<"0"> $x), (DiffeRet)),
                    (Select (FCmpOLT $x, $y), (ConstantFP<"0"> $y), (FNeg (DiffeRet)))
                  ],
                  (ForwardFromSummedReverse),
                  [ReadNone, NoUnwind]
                  >;

def : IntrPattern<(Op $x),
                  [["fabs"]],
                  [
                    (FMul (DiffeRet), (Select (FCmpOLT $x, (ConstantFP<"0"> $x)), (ConstantFP<"-1"> $x), (ConstantFP<"1"> $x)))
                  ],
                  (ForwardFromSummedReverse)
                  >;

def : IntrPattern<(Op $x),
                  [["sqrt"]],
                  [(Select (FCmpUEQ $x, (ConstantFP<"0"> $x)), (ConstantFP<"0"> $x), (FDiv (DiffeRet), (FMul (ConstantFP<"2"> $x), (Call<(SameFunc), [ReadNone,NoUnwind]> $x))))],
                  (ForwardFromSummedReverse)
                 >;


def : CallPattern<(Op (Op $x, $y):$z),
                  ["cmplx_sqrt"],
                  [(Select (And (FCmpUEQ $x, (ConstantFP<"0"> $x)), (FCmpUEQ $y, (ConstantFP<"0"> $y))), (ConstantCFP<"0", "0"> $z), (Conj (CFDiv (Conj (DiffeRet)), (CFMul (ConstantCFP<"2", "0"> $z), (Call<(SameFunc), [ReadNone,NoUnwind]> $z)))))],
                  (Select (And (FCmpUEQ $x, (ConstantFP<"0"> $x)), (FCmpUEQ $y, (ConstantFP<"0"> $y))), (ConstantCFP<"0", "0"> $z), (CFDiv (Shadow $z), (CFMul (ConstantCFP<"2", "0"> $z), (Call<(SameFunc), [ReadNone,NoUnwind]> $z))))
                 >;

def : IntrPattern<(Op $x, $y),
                  [["pow"]],
                  [
                    (CheckedMul (DiffeRet), (FMul $y, (Call<(SameFunc)> $x, (FSub $y, (ConstantFP<"1"> $y))))),
                    (CheckedMul (DiffeRet), (FMul (Call<(SameFunc)> $x, $y), (Intrinsic<"log"> $x)
                    ))
                  ],
                  (ForwardFromSummedReverse)
                 >;

def : IntrPattern<(Op $x, $y),
                  [["powi"]],
                  [
                    (Select 
                      (Or (ICmpEQ $y, (ConstantInt<0> $y) ), (And (GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ConstantInt::get(Type::getInt1Ty(gutils->newFunc->getContext()), gutils->strongZero)">), (FCmpOEQ (DiffeRet), (ConstantFP<"0"> $x) ) ) ),
                      (ConstantFP<"0"> $x),
                      (FMul (DiffeRet),
                        (FMul (SIToFP $y, (TypeOf $x)), (Call<(SameFunc)> $x, (ISub $y, (ConstantInt<1> $y))))
                      )
                     ),
                    (InactiveArg)
                  ],
                  (ForwardFromSummedReverse)
                 >;

def : IntrPattern<(Op $x, $y, $z),
                  [["fma"], ["fmuladd"]],
                  [
                    (FMul (DiffeRet), $y),
                    (FMul (DiffeRet), $x),
                    (DiffeRet)
                  ],
                  (ForwardFromSummedReverse)
                 >;

def : IntrPattern<(Op $x, $y),
                  [["experimental_vector_reduce_v2_fadd", "", "11"], ["vector_reduce_fadd", "12", ""]],
                  [
                    (DiffeRet),
                    (VectorSplat (VectorSize (TypeOf $y)), (DiffeRet))
                  ],
                  (Call<(SameFunc)> (SelectIfActive $x, (Shadow $x), (Zero $x)), (SelectIfActive $y, (Shadow $y), (Zero $y)))
                 >;

def : BinopPattern<(Op $x, $y), "FAdd", 0, 0,
                    [
                      (DiffeRet),
                      (DiffeRet),
                    ],
                    (ForwardFromSummedReverse)
                  >;
def : BinopPattern<(Op $x, $y), "FSub", 0, 0,
                    [
                      (DiffeRet),
                      (FNeg (DiffeRet)),
                    ],
                    (ForwardFromSummedReverse)
                  >;
def : BinopPattern<(Op $x, $y), "FMul", 0, 0,
                    [
                      (CheckedMul (DiffeRet), $y),
                      (CheckedMul (DiffeRet), $x)
                    ],
                    (ForwardFromSummedReverse)
                  >;
def : BinopPattern<(Op $x, $y), "FDiv", 0, 0,
                    [
                      (CheckedDiv (DiffeRet), $y),
                      (FNeg (CheckedMul (CheckedDiv (DiffeRet), $y), (FDiv $x, $y)))
                    ],
                    (CheckedDiv (FSub (SelectIfActive $x, (FMul (Shadow $x), $y), (Zero $x)), (SelectIfActive $y, (FMul (Shadow $y), $x), (Zero $y))), (FMul $y, $y))
                  >;

def : BinopPattern<(Op $x, $y), "FRem", 0, 0,
                    [
                      (DiffeRet),
                      (CheckedMul (DiffeRet), (FNeg (Intrinsic<"copysign"> (Intrinsic<"floor"> (Intrinsic<"fabs"> (FDiv $x, $y):$div)), $div)))
                    ],
                    (ForwardFromSummedReverse)
                  >;

def : InstPattern<(Op $x), "FNeg", 10, 0,
                    [
                      (FNeg (DiffeRet)),
                    ],
                    (ForwardFromSummedReverse)
                  >;
def : InstPattern<(Op $x), "Freeze", 10, 0,
                    [
                      (Freeze (DiffeRet)),
                    ],
                    (Freeze (Shadow $x))
                  >;

include "BlasDerivatives.td"                  

def : IntrPattern<(Op $x),
                  [["acos", "19", ""]],
                  [(FNeg (FDiv (DiffeRet), (Intrinsic<"sqrt"> (FSub (ConstantFP<"1.0"> $x), (FMul $x, $x)))))]  ,
                  (ForwardFromSummedReverse)                
                  >;

def : IntrPattern<(Op $x),
                  [["atan", "19", ""]],
                  [(FDiv (DiffeRet), (FAdd (FMul $x, $x), (ConstantFP<"1.0"> $x)))]  ,
                  (ForwardFromSummedReverse)                
                  >;

def : IntrPattern<(Op $x),
                  [["asin", "19", ""]],
                  [(FDiv (DiffeRet), (Intrinsic<"sqrt"> (FSub (ConstantFP<"1.0"> $x), (FMul $x, $x))))]  ,
                  (ForwardFromSummedReverse)                
                  >;
