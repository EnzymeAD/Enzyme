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

class CallOp<string mnemonic> {
  string name = mnemonic;
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

// Define rewrite pattern.
// def : CallPattern<(Op $y, $x), ["atan", "atanf", "atanl", "__fd_atan_1"], [(FMul (FAdd $y, $x), (DiffeRet<"">), (Shadow<""> $y) ), (FDiv $y, (ConstantFP<"1.0"> $y))]>;

def : CallPattern<(Op $x),
                  ["atan", "atanf", "atanl", "__fd_atan_1"],
                  [(FDiv (DiffeRet<"">), (FAdd (ConstantFP<"1.0"> $x), (FMul $x, $x)))]
                  >;

def : CallPattern<(Op $y, $x),
                  ["atan2"],
                  [
                  (FDiv (FMul (DiffeRet<"">), $x), (FAdd (FMul $x, $x), (FMul $y, $y))),
                  (FDiv (FNeg (FMul (DiffeRet<"">), $y)), (FAdd (FMul $x, $x), (FMul $y, $y))),
                  ]
                  >;
