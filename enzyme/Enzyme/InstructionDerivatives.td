class Type;
class CallPattern<dag patternToMatch, list<string> funcNames, list<dag> resultOps> {
  dag PatternToMatch = patternToMatch;
  list<string> names = funcNames;
  list<dag> ArgDerivatives = resultOps;
}

class Inst<string mnemonic> {
  string name = mnemonic;
}

// Define ops to rewrite.
def T1: Type;

def FDiv : Inst<"FDiv">;
def FAdd : Inst<"FAdd">;
def FSub : Inst<"FSub">;
def FMul : Inst<"FMul">;
def FNeg : Inst<"FNeg">;

def DifR : Inst<"DifR">;

// Same function as the one being called
def SameFunc {
}

// Same types / attributes / callingconv as original
// Different name, however
class SameTypesFunc<string name_> {
  string name = name_;
}

class Attribute<string name_> {
  string name = name_;
}
def ReadNone : Attribute<"ReadNone">;
def ReadOnly : Attribute<"ReadOnly">;
def NoUnwind : Attribute<"NoUnwind">;

// So we can also extend it by not input related 
// types, eg. for mem related Intrinsics
class TypeInfo {
}

class TypeOf<string val> : TypeInfo {
}

class Intrinsic<string name_, list<dag> types_, list<Attribute> _fnattrs=[]> {
  string name = name_;
  list<dag> types = types_;
  list<Attribute> fnattrs = _fnattrs;
}
class IntrinsicR<string name_, list<dag> types_> 
  : Intrinsic<name_, types_, !listconcat([ReadNone], _fnattrs)> {
}

class Call<dag mnemonic, list<Attribute> _fnattrs=[]> {
  dag func = mnemonic;
  list<Attribute> fnattrs = _fnattrs;
}
class CallRN<dag mnemonic, list<Attribute> _fnattrs=[]> 
  : Call<mnemonic, !listconcat([ReadNone,NoUnwind], _fnattrs)> {
}

def Op {
}

class ConstantFP<string val> {
  string value = val;
}

class DiffeRet<string val> {
}

class Shadow<string val> {
}

def : CallPattern<(Op $x),
                  ["atan", "atanf", "atanl", "__fd_atan_1"],
                  [(FDiv (DiffeRet<"">), (FAdd (ConstantFP<"1.0"> $x), (FMul $x, $x)))]
                  >;

def : CallPattern<(Op $y, $x),
                  ["atan2"],
                  [
                  (FDiv (FMul (DiffeRet<"">), $x), (FAdd (FMul $x, $x), (FMul $y, $y))),
                  (FNeg (FDiv (FMul (DiffeRet<"">), $y), (FAdd (FMul $x, $x), (FMul $y, $y))))
                  ]
                  >;
def : CallPattern<(Op $x),
                  ["cbrt"],
                  [(FDiv (FMul (DiffeRet<"">), (CallRN<(SameFunc)> $x) ), (FMul (ConstantFP<"3.0"> $x), $x))]
                  >;

def : CallPattern<(Op $x, $y),
                  ["hypot", "hypotf", "hypotl"],
                  [
                    (FDiv (FMul (DiffeRet<"">), $x), (CallRN<(SameFunc)> $x, $y)),
                    (FDiv (FMul (DiffeRet<"">), $y), (CallRN<(SameFunc)> $x, $y))
                  ]
                  >;

def : CallPattern<(Op $x),
                  ["tanh"],
                  [(FDiv (DiffeRet<"">), (FMul(CallRN<(SameTypesFunc<"cosh">)> $x):$c, $c))]>;

def : CallPattern<(Op $x),
                  ["tanhf"],
                  [(FDiv (DiffeRet<"">), (FMul(CallRN<(SameTypesFunc<"coshf">)> $x):$c, $c))]>;

def : CallPattern<(Op $x),
                  ["cosh"],
                  [(FMul (DiffeRet<"">), (CallRN<(SameTypesFunc<"sinh">)> $x))]>;
def : CallPattern<(Op $x),
                  ["coshf"],
                  [(FMul (DiffeRet<"">), (CallRN<(SameTypesFunc<"sinhf">)> $x))]>;

def : CallPattern<(Op $x),
                  ["sinh"],
                  [(FMul (DiffeRet<"">), (CallRN<(SameTypesFunc<"cosh">)> $x))]>;
def : CallPattern<(Op $x),
                  ["sinhf"],
                  [(FMul (DiffeRet<"">), (CallRN<(SameTypesFunc<"coshf">)> $x))]>;

def : CallPattern<(Op $x),
                  ["exp10"],
                  [(FMul (FMul (DiffeRet<"">), (CallRN<(SameFunc)> $x) ), (ConstantFP<"2.30258509299404568401799145468"> $x))]
                  >;
def : CallPattern<(Op $x),
                  ["tan", "tanf", "tanl"],
                  [(FMul (DiffeRet<"">), (FAdd (ConstantFP<"1.0"> $x), (FMul(CallRN<(SameFunc)> $x):$c, $c)))]>;
def : CallPattern<(Op $x, $y),
                  ["remainder"],
                  [
                    (DiffeRet<"">),
                    (FMul (FNeg (DiffeRet<"">)), (Intrinsic<"roundeven", [(TypeOf<""> $x)]> (FDiv $x, $y)))
                  ]
                  >;
def : CallPattern<(Op $x),
                  ["asin", "asinf", "asinl"],
                  [(FDiv (DiffeRet<"">), (Intrinsic<"sqrt", [(TypeOf<""> $x)]> (FSub (ConstantFP<"1.0"> $x), (FMul $x, $x)))  )]                  
                  >;
def : CallPattern<(Op $x),
                  ["log1p", "log1pf", "log1pl"],
                  [(FDiv (DiffeRet<"">), (FAdd (ConstantFP<"1.0"> $x), $x))]
                  >;
def : CallPattern<(Op $x),
                  ["expm1"],
                  [(FMul (Intrinsic<"exp", [(TypeOf<""> $x)]> $x), (DiffeRet<"">))]
                  >;
