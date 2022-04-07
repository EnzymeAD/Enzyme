class Type;
class CallPattern<dag patternToMatch, list<dag> resultOps> {
  dag PatternToMatch = patternToMatch;
  list<dag> ArgDerivatives = resultOps;
}

class Pat<dag pattern, dag result> : CallPattern<pattern, [result]>;
class Inst<string mnemonic> {
  string name = mnemonic;
}

// Define ops to rewrite.
def T1: Type;

def FDiv : Inst<"FDiv">;
def FAdd : Inst<"FAdd">;
def FSub : Inst<"FSub">;
def FMul : Inst<"FMul">;

def DifR : Inst<"DifR">;

class CallOp<string mnemonic> {
  string name = mnemonic;
}

def AtanOp : CallOp<"atan"> {
}

class ConstantFP<string val> {
  string value = val;
}

class DiffeRet<string val> {
}

class Shadow<string val> {
}

// Define rewrite pattern.
def : CallPattern<(AtanOp $y, $x), [(FMul (FAdd $y, $x), (DiffeRet<"">), (Shadow<""> $y) ), (FDiv $y, (ConstantFP<"1.0"> $y))]>;

