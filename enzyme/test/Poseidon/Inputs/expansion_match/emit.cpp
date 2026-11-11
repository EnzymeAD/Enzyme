// An opt pass plugin for expansion_reference_identity.cpp and
// expansion_model_identity.cpp, loaded after the Poseidon plugin, whose
// expansion emitters, materializer and accuracy model it calls.
//
// expansion-match-raw writes, for n = 2, 3, 4 components, raw-component
// functions p<n>_<op>(ptr a, [ptr b, [ptr c,]] ptr out) through the emitters,
// for op in add sub mul sqr div sqrt neg fma fmasq (fmasq: fma(a, a, c)), and
// p<n>_from_f64 / p<n>_to_f64 / p<n>_to_f32 through the boundary converters.
//
// expansion-match-model does the same, and also writes FP64 and FP32
// functions of operations (single operations and fixed random chains of three
// to six), evaluates the accuracy model on them as the pass does
// (getFPValues with a precision candidate whose changes are the function's
// operations) and then materializes them with applyExpansion, one call per
// change, as PTCandidate::apply does. Into $EXPANSION_MATCH_OUT it writes
// raw.bin (the raw-component cases and the model's results for them),
// app.bin (the points and the model's results for each function) and
// fns.inc (the functions, for the harness).
#include "Core/ExpansionModel.h"
#include "Model/Evaluators.h"
#include "Model/FPCore.h"
#include "Model/FPNode.h"
#include "Pass/Session.h"
#include "Techniques/Expansion.h"
#include "Techniques/Precision.h"
#include "Utils.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#if LLVM_VERSION_MAJOR >= 22 && !defined(ENZYME_LLVM22_PRERELEASE)
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif

#include "gen.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace llvm;
using namespace poseidon;

namespace {

const char *const kOps[] = {"add",  "sub", "mul", "sqr",  "div",
                            "sqrt", "neg", "fma", "fmasq"};
const unsigned kNOps = 9;
unsigned arity(StringRef op) {
  if (op == "sqr" || op == "sqrt" || op == "neg")
    return 1;
  return op == "fma" ? 3 : 2;
}

Function *makeFunction(Module &M, const Twine &name, Type *ret,
                       ArrayRef<Type *> args) {
  Function *F = Function::Create(FunctionType::get(ret, args, false),
                                 GlobalValue::ExternalLinkage, name, M);
  F->addFnAttr(Attribute::NoInline);
  BasicBlock::Create(M.getContext(), "entry", F);
  return F;
}

SmallVector<Value *, 4> load(IRBuilder<> &B, Value *p, unsigned n) {
  SmallVector<Value *, 4> v;
  for (unsigned i = 0; i < n; ++i)
    v.push_back(B.CreateLoad(
        B.getFloatTy(), B.CreateConstInBoundsGEP1_32(B.getFloatTy(), p, i)));
  return v;
}
void store(IRBuilder<> &B, Value *p, ArrayRef<Value *> v) {
  for (unsigned i = 0; i < v.size(); ++i)
    B.CreateStore(v[i], B.CreateConstInBoundsGEP1_32(B.getFloatTy(), p, i));
}

void emitRaw(Module &M, unsigned n, StringRef op) {
  LLVMContext &C = M.getContext();
  SmallVector<Type *, 4> args(arity(op) + 1, PointerType::get(C, 0));
  Function *F =
      makeFunction(M, "p" + Twine(n) + "_" + op, Type::getVoidTy(C), args);
  IRBuilder<> B(&F->getEntryBlock());
  SmallVector<SmallVector<Value *, 4>, 3> in;
  for (unsigned i = 0; i < arity(op); ++i)
    in.push_back(load(B, F->getArg(i), n));
  if (n == 2) {
    auto ds = [&](unsigned i) { return DSValue{in[i][0], in[i][1]}; };
    DSValue r{};
    if (op == "add")
      r = emitDSAdd(B, ds(0), ds(1));
    else if (op == "sub")
      r = emitDSSub(B, ds(0), ds(1));
    else if (op == "mul")
      r = emitDSMul(B, ds(0), ds(1));
    else if (op == "sqr")
      r = emitDSMul(B, ds(0), ds(0));
    else if (op == "div")
      r = emitDSDiv(B, ds(0), ds(1));
    else if (op == "sqrt")
      r = emitDSSqrt(B, ds(0));
    else if (op == "neg")
      r = emitDSNeg(B, ds(0));
    else if (op == "fma")
      r = emitDSFMA(B, ds(0), ds(1), ds(2));
    else
      r = emitDSFMA(B, ds(0), ds(0), ds(1));
    store(B, F->getArg(arity(op)), {r.hi, r.lo});
  } else {
    auto ev = [&](unsigned i) {
      ExpansionValue e;
      e.x.append(in[i].begin(), in[i].end());
      return e;
    };
    ExpansionValue r;
    if (op == "add")
      r = emitExpansionAdd(B, ev(0), ev(1));
    else if (op == "sub")
      r = emitExpansionSub(B, ev(0), ev(1));
    else if (op == "mul")
      r = emitExpansionMul(B, ev(0), ev(1));
    else if (op == "sqr")
      r = emitExpansionSqr(B, ev(0));
    else if (op == "div")
      r = emitExpansionDiv(B, ev(0), ev(1));
    else if (op == "sqrt")
      r = emitExpansionSqrt(B, ev(0));
    else if (op == "neg")
      r = emitExpansionNeg(B, ev(0));
    else if (op == "fma")
      r = emitExpansionAdd(B, emitExpansionMul(B, ev(0), ev(1)), ev(2));
    else
      r = emitExpansionAdd(B, emitExpansionSqr(B, ev(0)), ev(1));
    store(B, F->getArg(arity(op)), r.x);
  }
  B.CreateRetVoid();
}

void emitConversions(Module &M, unsigned n) {
  LLVMContext &C = M.getContext();
  Type *P = PointerType::get(C, 0);
  {
    Function *F = makeFunction(M, "p" + Twine(n) + "_from_f64",
                               Type::getVoidTy(C), {Type::getDoubleTy(C), P});
    IRBuilder<> B(&F->getEntryBlock());
    if (n == 2) {
      DSValue d = emitF64ToDS(B, F->getArg(0));
      store(B, F->getArg(1), {d.hi, d.lo});
    } else {
      store(B, F->getArg(1), emitToExpansion(B, F->getArg(0), n).x);
    }
    B.CreateRetVoid();
  }
  for (Type *T : {Type::getDoubleTy(C), Type::getFloatTy(C)}) {
    Function *F = makeFunction(
        M, "p" + Twine(n) + (T->isDoubleTy() ? "_to_f64" : "_to_f32"), T, {P});
    IRBuilder<> B(&F->getEntryBlock());
    auto l = load(B, F->getArg(0), n);
    ExpansionValue e;
    e.x.append(l.begin(), l.end());
    B.CreateRet(n == 2 ? emitDSToFP(B, DSValue{l[0], l[1]}, T)
                       : emitExpansionToFP(B, e, T));
  }
}

void emitAllRaw(Module &M) {
  for (unsigned n : {2u, 3u, 4u}) {
    for (const char *op : kOps)
      emitRaw(M, n, op);
    emitConversions(M, n);
  }
}

// A function of three arguments and the operations to convert, grouped into
// the changes of one precision candidate.
struct AppFn {
  Function *F;
  unsigned n;
  const char *kind;
  SmallVector<SmallVector<Instruction *, 8>, 2> changes;
};

Value *intrinsic(IRBuilder<> &B, Intrinsic::ID id, ArrayRef<Value *> args) {
  return B.CreateIntrinsic(id, {args[0]->getType()}, args);
}

bool isFmuladd(Instruction *I) {
  auto *II = dyn_cast<IntrinsicInst>(I);
  return II && II->getIntrinsicID() == Intrinsic::fmuladd;
}

// A chain of `len` operations: each reads the previous result, the other
// operands are arguments, earlier results or constants, so values repeat
// (squares, fma(x, x, c)) and results have several uses. The operations to
// convert are every FP operation but the fabs that keeps a square root real.
Value *chain(IRBuilder<> &B, Function *F, std::mt19937_64 &rng, unsigned len,
             SmallVectorImpl<Instruction *> &ops) {
  Type *T = F->getReturnType();
  SmallVector<Value *, 12> vals = {F->getArg(0), F->getArg(1), F->getArg(2)};
  const double consts[] = {3.14159265358979323846, 0.1, 1.0 / 3.0, -2.5, 1e-3};
  auto other = [&]() -> Value * {
    if (rng() % 8 == 0)
      return ConstantFP::get(T, consts[rng() % 5]);
    return vals[rng() % vals.size()];
  };
  Value *last = nullptr;
  for (unsigned i = 0; i < len; ++i) {
    Value *cur = last ? last : vals[rng() % 3];
    Value *x = cur, *y = other();
    if (rng() % 2)
      std::swap(x, y);
    Value *r = nullptr;
    switch (rng() % 10) {
    case 0:
      r = B.CreateFAdd(x, y);
      break;
    case 1:
      r = B.CreateFSub(x, y);
      break;
    case 2:
      r = B.CreateFMul(x, y);
      break;
    case 3:
      r = B.CreateFMul(cur, cur);
      break;
    case 4:
      r = B.CreateFDiv(x, y);
      break;
    case 5:
      r = intrinsic(B, Intrinsic::sqrt, {intrinsic(B, Intrinsic::fabs, {cur})});
      break;
    case 6:
      r = B.CreateFNeg(cur);
      break;
    case 7:
      r = intrinsic(B, Intrinsic::fmuladd, {x, y, other()});
      break;
    case 8:
      r = intrinsic(B, Intrinsic::fmuladd, {cur, cur, y});
      break;
    case 9:
      r = B.CreateFAdd(B.CreateFMul(x, y), other());
      break;
    }
    vals.push_back(r);
    last = r;
  }
  for (Instruction &I : F->getEntryBlock()) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (I.getType() == T && !(II && II->getIntrinsicID() == Intrinsic::fabs))
      ops.push_back(&I);
  }
  return last;
}

std::vector<AppFn> emitApplied(Module &M) {
  LLVMContext &C = M.getContext();
  std::vector<AppFn> fns;
  using A = ArrayRef<Value *>;
  auto single = [&](unsigned n, StringRef name,
                    function_ref<Value *(IRBuilder<> &, A)> body) {
    Type *D = Type::getDoubleTy(C);
    Function *F = makeFunction(M, "a" + Twine(n) + "_" + name, D, {D, D, D});
    IRBuilder<> B(&F->getEntryBlock());
    B.CreateRet(body(B, {F->getArg(0), F->getArg(1), F->getArg(2)}));
    AppFn a{F, n, "single operations", {{}}};
    for (Instruction &I : F->getEntryBlock())
      if (I.getType()->isDoubleTy())
        a.changes[0].push_back(&I);
    fns.push_back(a);
  };
  std::mt19937_64 rng(0xc4a1257);
  for (unsigned n : {2u, 3u, 4u}) {
    single(n, "add",
           [](IRBuilder<> &B, A a) { return B.CreateFAdd(a[0], a[1]); });
    single(n, "sub",
           [](IRBuilder<> &B, A a) { return B.CreateFSub(a[0], a[1]); });
    single(n, "mul",
           [](IRBuilder<> &B, A a) { return B.CreateFMul(a[0], a[1]); });
    single(n, "sq",
           [](IRBuilder<> &B, A a) { return B.CreateFMul(a[0], a[0]); });
    single(n, "div",
           [](IRBuilder<> &B, A a) { return B.CreateFDiv(a[0], a[1]); });
    single(n, "sqrt", [](IRBuilder<> &B, A a) {
      return intrinsic(B, Intrinsic::sqrt, {a[0]});
    });
    single(n, "neg", [](IRBuilder<> &B, A a) { return B.CreateFNeg(a[0]); });
    single(n, "fma", [](IRBuilder<> &B, A a) {
      return intrinsic(B, Intrinsic::fmuladd, {a[0], a[1], a[2]});
    });
    single(n, "fmasq", [](IRBuilder<> &B, A a) {
      return intrinsic(B, Intrinsic::fmuladd, {a[0], a[0], a[1]});
    });
    single(n, "mulpi", [](IRBuilder<> &B, A a) {
      return B.CreateFMul(
          a[0], ConstantFP::get(B.getDoubleTy(), 3.14159265358979323846));
    });
    // Chain kinds by k % 4: one change; two changes (the first half, then the
    // rest); one operation left in FP64 (not an fmuladd, which the model
    // evaluates fused and a host without FMA does not); an FP32 function.
    for (unsigned k = 0; k < 16; ++k) {
      const unsigned kind = k % 4;
      Type *T = kind == 3 ? Type::getFloatTy(C) : Type::getDoubleTy(C);
      Function *F = makeFunction(
          M, (kind == 3 ? "f" : "c") + Twine(n) + "_" + Twine(k), T, {T, T, T});
      IRBuilder<> B(&F->getEntryBlock());
      SmallVector<Instruction *, 16> ops;
      B.CreateRet(chain(B, F, rng, 3 + rng() % 4, ops));
      static const char *const kinds[] = {
          "chains, one change", "chains, two changes",
          "chains, one operation left in FP64", "FP32 chains"};
      AppFn a{F, n, kinds[kind], {}};
      if (kind == 1) {
        a.changes.emplace_back(ops.begin(), ops.begin() + ops.size() / 2);
        a.changes.emplace_back(ops.begin() + ops.size() / 2, ops.end());
      } else {
        a.changes.emplace_back(ops.begin(), ops.end());
        if (kind == 2) {
          SmallVector<Instruction *, 8> skippable;
          for (Instruction *I : ops)
            if (I != ops.back() && !isFmuladd(I))
              skippable.push_back(I);
          if (!skippable.empty())
            erase(a.changes[0], skippable[rng() % skippable.size()]);
        }
      }
      fns.push_back(a);
    }
  }
  return fns;
}

// The accuracy model's graph of F, built as buildSubgraphs builds it.
std::shared_ptr<FPNode>
nodeOf(Value *V, std::unordered_map<Value *, std::shared_ptr<FPNode>> &m) {
  if (auto it = m.find(V); it != m.end())
    return it->second;
  std::string dtype = V->getType()->isFloatTy() ? "f32" : "f64";
  std::shared_ptr<FPNode> node;
  if (auto *A = dyn_cast<Argument>(V)) {
    node = std::make_shared<FPLLValue>(A, "__arg", dtype);
  } else if (auto *CF = dyn_cast<ConstantFP>(V)) {
    SmallString<16> s;
    CF->getValueAPF().toString(s);
    node = std::make_shared<FPConst>(s.c_str(), dtype);
  } else {
    auto *I = cast<Instruction>(V);
    node = std::make_shared<FPLLValue>(I, getHerbieOperator(*I), dtype);
    for (Value *op : I->operands())
      if (!isa<Function>(op))
        node->addOperand(nodeOf(op, m));
  }
  m[V] = node;
  return node;
}

PrecisionChangeType expansionType(unsigned n) {
  return n == 2   ? PrecisionChangeType::Expansion2
         : n == 3 ? PrecisionChangeType::Expansion3
                  : PrecisionChangeType::Expansion4;
}

std::string outPath(const char *file) {
  const char *d = getenv("EXPANSION_MATCH_OUT");
  if (!d)
    report_fatal_error("expansion-match-model: set EXPANSION_MATCH_OUT");
  return std::string(d) + "/" + file;
}

void modelRaw() {
  FILE *out = fopen(outPath("raw.bin").c_str(), "wb");
  for (unsigned n : {2u, 3u, 4u})
    expansion_match::forEachCase(n, 10000, [&](const expansion_match::Case &k) {
      ExpansionLimbs a = ExpansionLimbs::of(k.a, n),
                     b = ExpansionLimbs::of(k.b, n),
                     c = ExpansionLimbs::of(k.c, n);
      ExpansionLimbs r[kNOps] = {
          expansionAdd(a, b), expansionSub(a, b),    expansionMul(a, b),
          expansionSqr(a),    expansionDiv(a, b),    expansionSqrt(a),
          expansionNeg(a),    expansionFma(a, b, c), expansionFmaSqr(a, c)};
      fwrite(&k, sizeof k, 1, out);
      for (auto &x : r)
        fwrite(x.x, 4, 4, out);
      double v = (double)k.a[0] + (double)k.a[1];
      fwrite(expansionFromF64(v, n).x, 4, 4, out);
      double d = expansionToF64(a);
      float f = expansionToF32(a);
      fwrite(&d, 8, 1, out);
      fwrite(&f, 4, 1, out);
    });
  fclose(out);
}

void modelApplied(std::vector<AppFn> &fns) {
  // Points: mostly moderate exponents, some over the whole range, some
  // cancelling, some signed zeros and some FP32-subnormal values.
  std::vector<double> pts;
  std::mt19937_64 rng(0x9e3779b9);
  for (int i = 0; i < 4000; ++i)
    for (int j = 0; j < 3; ++j) {
      int e = i % 10 == 0 ? (int)(rng() % 2046) - 1022 : (int)(rng() % 61) - 30;
      double v = std::ldexp(1.0 + (double)(rng() >> 11) * 0x1p-53, e);
      if (i % 101 == j)
        v = std::ldexp(1.0 + (double)(rng() >> 11) * 0x1p-53, -140);
      pts.push_back(rng() & 1 ? -v : v);
    }
  for (int i = 0; i < 4000; i += 50)
    pts[3 * i + 1] = -pts[3 * i] * (1 + 0x1p-40);
  for (int i = 0; i < 4000; i += 97)
    pts[3 * i + i % 3] = i % 2 ? 0.0 : -0.0;
  FILE *out = fopen(outPath("app.bin").c_str(), "wb");
  FILE *inc = fopen(outPath("fns.inc").c_str(), "w");
  fwrite(pts.data(), 8, pts.size(), out);
  for (AppFn &a : fns) {
    std::unordered_map<Value *, std::shared_ptr<FPNode>> m;
    std::shared_ptr<FPNode> root =
        nodeOf(a.F->getEntryBlock().getTerminator()->getOperand(0), m);
    const bool f32 = a.F->getReturnType()->isFloatTy();
    SmallVector<PrecisionChange> changes;
    for (auto &ch : a.changes) {
      SetVector<FPLLValue *> nodes;
      for (Instruction *I : ch)
        nodes.insert(cast<FPLLValue>(m.at(I).get()));
      changes.emplace_back(
          nodes, f32 ? PrecisionChangeType::FP32 : PrecisionChangeType::FP64,
          expansionType(a.n));
    }
    PTCandidate pt(changes, "expansion-match");
    for (size_t i = 0; i < pts.size() / 3; ++i) {
      MapVector<Value *, double> in;
      for (unsigned j = 0; j < 3; ++j)
        in[a.F->getArg(j)] =
            f32 ? (double)(float)pts[3 * i + j] : pts[3 * i + j];
      SmallVector<double, 1> r;
      getFPValues({root.get()}, in, r, pt.changes);
      fwrite(&r[0], 8, 1, out);
    }
    fprintf(inc, "FN(%s, %u, %s, \"%s\")\n", a.F->getName().str().c_str(), a.n,
            f32 ? "float" : "double", a.kind);
  }
  fclose(out);
  fclose(inc);
}

void materialize(Session &S, std::vector<AppFn> &fns) {
  for (AppFn &a : fns)
    for (auto &ch : a.changes) {
      SmallPtrSet<Instruction *, 16> all(ch.begin(), ch.end());
      DenseMap<Value *, Value *> restored;
      if (a.n == 2)
        applyExpansion(S, ch, all, &restored);
      else
        applyExpansion(S, a.n, ch, all, &restored);
    }
}

struct EmitRaw : PassParent<EmitRaw> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    emitAllRaw(M);
    return PreservedAnalyses::none();
  }
};

struct EmitModel : PassParent<EmitModel> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    emitAllRaw(M);
    modelRaw();
    std::vector<AppFn> fns = emitApplied(M);
    modelApplied(fns);
    materialize(sessionOf(M, MAM), fns);
    return PreservedAnalyses::none();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ExpansionMatch", "0", [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (name == "expansion-match-raw") {
                    MPM.addPass(EmitRaw());
                    return true;
                  }
                  if (name == "expansion-match-model") {
                    MPM.addPass(EmitModel());
                    return true;
                  }
                  return false;
                });
          }};
}
