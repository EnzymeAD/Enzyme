#ifndef ENZYME_MLIR_IMPLEMENTATIONS_COMMON
#define ENZYME_MLIR_IMPLEMENTATIONS_COMMON

class InactiveOp<string dialect_, string opName_> {
  string dialect = dialect_;
  string opName = opName_;
}

class AllocationOp<string dialect_, string opName_> {
  string dialect = dialect_;
  string opName = opName_;
}

class ControlFlowOp<string dialect_, string opName_, string impl_> {
  string dialect = dialect_;
  string opName = opName_;
  string impl = impl_;
}

class InactiveArgSpec {
  bit asserting;
}
def InactiveArg : InactiveArgSpec {
  bit asserting = 0;
}
def AssertingInactiveArg : InactiveArgSpec {
  bit asserting = 1;
}


def Unimplemented {

}

class MemoryIdentityOp<string dialect_, string opName_, list<int> ptrargs_, list<int> storedargs_ = [], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []> {
  string dialect = dialect_;
  string opName = opName_;
  dag PatternToMatch = patternToMatch;
  list<int> ptrargs = ptrargs_;
  list<int> storedargs = storedargs_;
  list<dag> reverse = reverse_;
}

class ReadOnlyIdentityOp<string dialect_, string opName_, list<int> ptrargs_, dag patternToMatch=(Unimplemented), list<dag> reverse_ = []> : MemoryIdentityOp<dialect_, opName_, ptrargs_, [], patternToMatch, reverse_>;

class ReturnOp<string dialect_, string opName_> {
  string dialect = dialect_;
  string opName = opName_;
}

class BranchOp<string dialect_, string opName_> {
  string dialect = dialect_;
  string opName = opName_;
}

class RegionTerminatorOp<string dialect_, string opName_> {
  string dialect = dialect_;
  string opName = opName_;
}

class ForwardFromSummedReverseInternal<int unused_> {
  int unused = unused_;
}
def ForwardFromSummedReverse : ForwardFromSummedReverseInternal<0>;


class MLIRDerivative<string dialect_, string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> {
  string dialect = dialect_;
  string opName = opName_;
  dag PatternToMatch = patternToMatch;
  list<dag> ArgDerivatives = resultOps;
  dag ArgDuals = forwardOps;
}

class Operation<bit usesPrimal_, bit usesShadow_, bit usesCustom_=0> {
  bit usesPrimal = usesPrimal_;
  bit usesShadow = usesShadow_;
  bit usesCustom = usesCustom_;
}

class DiffeRetIndex<list<int> indices_> {
  list<int> indices = indices_;
}
def DiffeRet : DiffeRetIndex<[-1]>;

def Shadow : Operation</*primal*/0, /*shadow*/1> {
}

class GlobalExpr<bit uses_primal, bit uses_shadow, string val> : Operation<uses_primal, uses_shadow>{
  string value = val;
}

class Inst<string mnemonic, string dialect_, string postop_="">  : Operation</*primal*/1, /*shadow*/0> {
  string name = mnemonic;
  string dialect = dialect_;
  string postop = postop_;
}

def Op {
}

def SelectIfActive : Operation</*primal*/0, /*shadow*/0, /*custom*/1> {

}

def SelectIfComplex : Operation</*primal*/0, /*shadow*/0, /*custom*/1> {

}

class ConstantFP<string val, string dialect_, string op_, string type_=""> : Operation</*primal*/0, /*shadow*/0> {
  string value = val;
  string dialect = dialect_;
  string opName = op_;
  string type = type_;
}

class ConjIfComplex<string dialect_, string op_> : Operation</*primal*/1, /*shadow*/0> {
  string dialect = dialect_;
  string opName = op_;
}

def ResultTypes : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "op->getResultTypes()">;

def TypeOf : Operation</*primal*/0, /*shadow*/0> {
}

class ComplexInst<string m> : Inst<m, "complex">;
class ArithInst<string m> : Inst<m, "arith">;
class MathInst<string m> : Inst<m, "math">;

def AddF : ArithInst<"AddFOp">;
def SubF : ArithInst<"SubFOp">;
def NegF : ArithInst<"NegFOp">;
def MulF : ArithInst<"MulFOp">;
def DivF : ArithInst<"DivFOp">;
def RemF : ArithInst<"RemFOp">;

def CheckedMulF : ArithInst<"MulFOp">;
def CheckedDivF : ArithInst<"DivFOp">;

def CosF : MathInst<"CosOp">;
def SinF : MathInst<"SinOp">;
def ExpF : MathInst<"ExpOp">;
def SqrtF : MathInst<"SqrtOp">;

#endif // ENZYME_MLIR_IMPLEMENTATIONS_COMMON
