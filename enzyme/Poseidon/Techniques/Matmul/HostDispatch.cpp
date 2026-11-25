//===- HostDispatch.cpp - host-side GEMM library dispatch -----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// See HostDispatch.h.
//
//===----------------------------------------------------------------------===//

#include "HostDispatch.h"
#include "Enzyme/Utils.h"
#include "IR/Utils.h"
#include "MatmulInternal.h"
#include "Pass/Session.h"
#include "Sites/Annotations.h"
#include "Sites/LaunchDescriptors.h"
#include "Sites/SiteIdentity.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"
#include "runtime/GemmDispatch.inc"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <map>
#include <optional>
#include <string>

using namespace llvm;

namespace poseidon {

std::string runtimeDimString(const RuntimeDim &d) {
  if (d.isConst())
    return std::to_string(d.add);
  std::string s;
  if (d.mul != 1)
    s += std::to_string(d.mul) + "*";
  for (const LaunchRef &f : d.factors) {
    if (&f != &d.factors.front())
      s += "*";
    s += launchRefString(f, "arg");
  }
  if (d.add)
    s += (d.add > 0 ? "+" : "") + std::to_string(d.add);
  return s;
}

// Replaces the GEMM result where the epilogue consumes it with a load from the
// output buffer that the prepended host dispatch fills, so the reduction loop
// dies. F stores a value that reaches the product to an argument, as it does
// when its constant-shape note is not standalone.
static void fissionGemmForDispatch(const Options &O, Function &F,
                                   const AbstractMatmul &m) {
  StoreInst *cStore = productStore(F, m);
  assert(cStore &&
         "a body with a fused note stores the product to an argument");

  // Replace the GEMM result where the epilogue consumes it with a load from the
  // output buffer (filled by the prepended host dispatch); the reduction loop
  // becomes dead. In-loop (recurrence) uses stay.
  Value *resultVal = cStore->getValueOperand();
  Value *outAddr = cStore->getPointerOperand();
  IRBuilder<> B(cStore);
  LoadInst *loaded =
      B.CreateLoad(resultVal->getType(), outAddr, "dispatch.gemm");
  SmallVector<Use *, 8> repl;
  for (Use &U : resultVal->uses()) {
    User *user = U.getUser();
    if (user == cStore)
      continue;
    if (auto *I = dyn_cast<Instruction>(user))
      if (m.footprint.count(I))
        continue; // in-loop recurrence, keep
    repl.push_back(&U);
  }
  for (Use *U : repl)
    U->set(loaded);
  cStore->eraseFromParent(); // dispatch already wrote the output buffer
  if (O.Diagnostics.Print)
    errs() << "[poseidon] fissioned the product in " << F.getName()
           << ": the epilogue reads the library call's output and the "
              "reduction loop is left to dead-code elimination\n";
}

// The launch argument each caller value is, -1 for one that is not: the host
// reads the argument itself, so a pointer may differ from it only by casts.
static std::vector<int> launchArgsOf(ArrayRef<Value *> primalArgs) {
  std::vector<int> paramToArg(primalArgs.size(), -1);
  for (size_t i = 0; i < primalArgs.size(); ++i) {
    Value *v = primalArgs[i];
    if (v && v->getType()->isPointerTy())
      v = v->stripPointerCasts();
    if (auto *A = dyn_cast_or_null<Argument>(v))
      paramToArg[i] = (int)A->getArgNo();
  }
  return paramToArg;
}

// Whether `wrapper` does nothing with an effect but call its site, so that a
// library call can take the place of its whole launch.
static bool onlyCallsItsSite(const Function &wrapper) {
  unsigned effects = 0;
  for (const Instruction &I : instructions(wrapper)) {
    if (!I.mayHaveSideEffects())
      continue;
    if (const auto *II = dyn_cast<IntrinsicInst>(&I);
        II && II->isAssumeLikeIntrinsic())
      continue;
    ++effects;
  }
  return effects <= 1;
}

const CallBase *siteCallOf(const Session &S, const Function &F) {
  SmallVector<Value *, 8> callerArgs;
  const Function *kernel = nullptr;
  for (unsigned i = 0, e = F.arg_size(); i < e; ++i) {
    Value *v = siteArg(S.Sites, &F, i);
    callerArgs.push_back(v);
    if (auto *A = dyn_cast_or_null<Argument>(v))
      kernel = A->getParent();
    else if (auto *I = dyn_cast_or_null<Instruction>(v))
      kernel = I->getFunction();
  }
  if (!kernel)
    return nullptr;
  for (const Instruction &I : instructions(*kernel))
    if (const auto *CB = dyn_cast<CallBase>(&I);
        CB && llvm::all_of(callerArgs, [&](Value *v) {
          return !v || llvm::is_contained(CB->args(), v);
        }))
      return CB;
  return nullptr;
}

Expected<std::optional<GemmBodyNote>> libraryCallNote(const Session &S,
                                                      const AbstractMatmul &m) {
  Function *F = const_cast<Function *>(anchorOf(m).getFunction());
  GemmBodyNote note;
  switch (m.origin) {
  case AbstractMatmul::Origin::HostGemmLoopNest:
    note = m.hostGemm->library;
    break;
  case AbstractMatmul::Origin::ScalarLoopReduction: {
    Expected<std::optional<GemmBodyNote>> n =
        constantShapeLibraryNote(S.Opts, *F, m);
    if (!n)
      return n.takeError();
    if (!*n)
      return std::nullopt;
    note = **n;
    break;
  }
  }
  auto refuse = [&](const Twine &why) -> std::optional<GemmBodyNote> {
    if (S.Opts.Diagnostics.Print)
      llvm::errs() << "[poseidon] " << F->getName() << ": " << why << '\n';
    return std::nullopt;
  };
  // The host replaces or precedes the whole launch of the kernel with the
  // library call, reading every operand and dimension from that launch.
  const CallBase *site = siteCallOf(S, *F);
  if (!site)
    return refuse("the kernel the site is called from is not known");
  const Function *kernel = site->getFunction();
  if (!onlyCallsItsSite(*kernel))
    return refuse("its kernel " + kernel->getName() +
                  " does more than call this site, and the library call takes "
                  "the place of the whole launch");
  SmallVector<Value *, 8> callerArgs;
  for (unsigned i = 0, e = F->arg_size(); i < e; ++i)
    callerArgs.push_back(siteArg(S.Sites, F, i));
  const std::vector<int> paramToArg = launchArgsOf(callerArgs);
  auto mapped = [&](int param) {
    return param >= 0 && (size_t)param < paramToArg.size() &&
           paramToArg[param] >= 0;
  };
  SmallVector<int, 16> params = {note.cParam, note.aParam, note.bParam};
  for (const RuntimeDim &d : note.dims)
    for (const LaunchRef &f : d.factors)
      params.push_back(f.param);
  for (const auto &[x, y] : note.assumedEqual)
    params.append({x.param, y.param});
  if (!llvm::all_of(params, mapped))
    return refuse("an operand or dimension of the product is not an argument "
                  "of the kernel " +
                  kernel->getName() + ", from which the host reads it");

  return note;
}

Error applyLibraryCall(Session &S, const AbstractMatmul &m,
                       const GemmBodyNote &note, const SchemeVariant &v) {
  assert(S.Opts.Matmul.OzakiHostDispatch &&
         "library calls are proposed only under -poseidon-ozaki-host-dispatch");
  Function *F = const_cast<Function *>(anchorOf(m).getFunction());
  // One descriptor per body: the host replaces or precedes its kernel's launch
  // with one library call, and the solve chooses for the body's products
  // independently.
  if (S.Matmul.GemmBodies.count(F))
    return userError(anchorOf(m),
                     "a second matrix product of " + F->getName() +
                         " was selected to run as a library call, and a "
                         "kernel's launch takes one; leave out "
                         "-poseidon-ozaki-host-dispatch for this body.");
  S.Matmul.GemmBodies[F] = {note, v};
  if (!note.standalone)
    fissionGemmForDispatch(S.Opts, *F, m);
  return Error::success();
}

static void emitDescriptor(Session &S, StringRef wrapperName, int cArg,
                           int aArg, int bArg, const GemmBodyNote &n,
                           const SchemeVariant &v) {
  const unsigned modulusField = descriptorModulusField(v);
  std::string message;
  raw_string_ostream msg(message);
  msg << "[poseidon] wrote descriptor for " << wrapperName
      << ": C=" << launchRefString({cArg, n.cOffset}, "arg")
      << " A=" << launchRefString({aArg, n.aOffset}, "arg")
      << " B=" << launchRefString({bArg, n.bOffset}, "arg") << ' '
      << runtimeDimString(n.dims[DimM]) << 'x'
      << runtimeDimString(n.dims[DimNcols]) << 'x'
      << runtimeDimString(n.dims[DimK]) << " nm=" << modulusField
      << (n.standalone ? " (standalone)" : " (fused)");
  for (const auto &[x, y] : n.assumedEqual)
    msg << ", if " << launchRefString(x, "arg")
        << " == " << launchRefString(y, "arg");
  msg << '\n';
  recordDescriptor(
      S.Launches, LaunchKind::HostDispatch, wrapperName,
      [&](raw_ostream &os) {
        os << ' ' << cArg << ' ' << aArg << ' ' << bArg << ' '
           << (n.standalone ? 1 : 0) << ' ' << modulusField << ' '
           << (n.aColMajor ? 1 : 0) << ' ' << (n.bColMajor ? 1 : 0) << ' '
           << (n.cColMajor ? 1 : 0) << ' ' << (unsigned)v.scheme;
        // Per dimension the factor count, each factor's argument and offset,
        // mul and add.
        os << " rd";
        for (const RuntimeDim &x : n.dims) {
          os << ' ' << x.factors.size();
          for (const LaunchRef &f : x.factors)
            os << ' ' << f.param << ' ' << f.offset;
          os << ' ' << x.mul << ' ' << x.add;
        }
        os << ' ' << n.beta;
        // The extents the kernel's index guard gates its store with,
        // intersected at the dispatch with the launch geometry.
        if (n.launchCrop)
          os << " lc " << n.cropMAxis << ' ' << n.cropNAxis;
        if (n.cOffset >= 0 || n.aOffset >= 0 || n.bOffset >= 0)
          os << " pf " << n.cOffset << ' ' << n.aOffset << ' ' << n.bOffset;
        if (!n.assumedEqual.empty()) {
          os << " eq " << n.assumedEqual.size();
          for (const auto &[x, y] : n.assumedEqual)
            os << ' ' << x.param << ' ' << x.offset << ' ' << y.param << ' '
               << y.offset;
        }
      },
      std::move(message));
}

// Shared descriptor emit for the per-site path and the deferred joint-dp
// flush: map C/A/B and the dimensions through `paramToArg` (body parameter to
// wrapper launch argument), which libraryCallNote checked.
static void emitMappedDescriptor(Session &S, Function &wrapper,
                                 const ChosenLibraryCall &call,
                                 ArrayRef<int> paramToArg) {
  const GemmBodyNote &n = call.note;
  auto mapParam = [&](int bp) -> int {
    if (bp < 0 || (size_t)bp >= paramToArg.size())
      return -1;
    return paramToArg[bp];
  };
  int cArg = mapParam(n.cParam);
  int aArg = mapParam(n.aParam);
  int bArg = mapParam(n.bParam);
  assert(cArg >= 0 && aArg >= 0 && bArg >= 0 &&
         "a library call is proposed only with its operands kernel arguments");
  // Runtime dimensions name body parameters too, and the host side only sees
  // the wrapper's launch arguments, so they go through the same mapping.
  GemmBodyNote mapped = n;
  for (RuntimeDim &x : mapped.dims)
    for (LaunchRef &f : x.factors)
      f.param = mapParam(f.param);
  for (auto &[x, y] : mapped.assumedEqual) {
    x.param = mapParam(x.param);
    y.param = mapParam(y.param);
  }
  assert(onlyCallsItsSite(wrapper) &&
         "a library call is proposed only for a kernel that does nothing else");
  emitDescriptor(S, wrapper.getName(), cArg, aArg, bArg, mapped, call.variant);
}

void writeGemmDescriptor(Session &S, Function &wrapper,
                         ArrayRef<Value *> primalArgs,
                         const ChosenLibraryCall &call) {
  emitMappedDescriptor(S, wrapper, call, launchArgsOf(primalArgs));
}

void addPendingGemmDispatch(Session &S, const Function *body, Function &wrapper,
                            ArrayRef<Value *> primalArgs) {
  S.Matmul.PendingGemms[body] = {wrapper.getName().str(),
                                 launchArgsOf(primalArgs)};
}

void flushPendingGemmDispatches(Session &S) {
  for (auto &[body, pg] : S.Matmul.PendingGemms)
    // A site not materialized as a host dispatch has no note.
    if (auto n = S.Matmul.GemmBodies.find(body); n != S.Matmul.GemmBodies.end())
      emitMappedDescriptor(S, *body->getParent()->getFunction(pg.wrapperName),
                           n->second, pg.paramToArg);
  S.Matmul.PendingGemms.clear();
}

namespace {
// A descriptor read back on the host side, its operand roles and dimension
// factors naming launch arguments of `kernel`.
struct LaunchDesc : GemmBodyNote {
  std::string kernel;
  /// The runtime entry and its last argument, as the descriptor records them.
  Scheme scheme;
  unsigned modulusField;
};
} // namespace

// The entry points of runtime/GemmDispatch.inc.
static const char *dispatchSym(Scheme s) {
  switch (s) {
#define POSEIDON_GEMM_ENTRY(Entry, Symbol)                                     \
  case Scheme::Entry:                                                          \
    return #Symbol;
#define POSEIDON_GEMM_ENTRY_MODE(Entry, Symbol, Mode)                          \
  POSEIDON_GEMM_ENTRY(Entry, Symbol)
#include "runtime/GemmDispatch.inc"
  }
  llvm_unreachable("unknown Scheme");
}

// Whether the entry takes a last Mode parameter, the descriptor's modulus
// field.
static bool dispatchHasMode(Scheme s) {
  switch (s) {
#define POSEIDON_GEMM_ENTRY(Entry, Symbol)                                     \
  case Scheme::Entry:                                                          \
    return false;
#define POSEIDON_GEMM_ENTRY_MODE(Entry, Symbol, Mode)                          \
  case Scheme::Entry:                                                          \
    return true;
#include "runtime/GemmDispatch.inc"
  }
  llvm_unreachable("unknown Scheme");
}

static FunctionCallee getDispatchCallee(Module &M, Scheme scheme) {
  LLVMContext &C = M.getContext();
  Type *Ptr = getInt8PtrTy(C);
  Type *I32 = Type::getInt32Ty(C);
  Type *Double = Type::getDoubleTy(C);
#define POSEIDON_GEMM_TYPE(CType, IRType, Name) IRType,
  SmallVector<Type *, 16> params = {
      POSEIDON_GEMM_PARAMS(POSEIDON_GEMM_TYPE, POSEIDON_GEMM_TYPE)};
#undef POSEIDON_GEMM_TYPE
  if (dispatchHasMode(scheme))
    params.push_back(I32);
  return M.getOrInsertFunction(
      dispatchSym(scheme),
      FunctionType::get(Type::getVoidTy(C), params, false));
}

namespace {
// The arguments of an entry by the names of GemmDispatch.inc, which values()
// puts in its order: a parameter the table renames does not compile here.
struct DispatchArgs {
#define POSEIDON_GEMM_FIELD(CType, IRType, Name) Value *Name = nullptr;
  POSEIDON_GEMM_PARAMS(POSEIDON_GEMM_FIELD, POSEIDON_GEMM_FIELD)
#undef POSEIDON_GEMM_FIELD

  SmallVector<Value *, 16> values() const {
#define POSEIDON_GEMM_VALUE(CType, IRType, Name) Name,
    SmallVector<Value *, 16> v = {
        POSEIDON_GEMM_PARAMS(POSEIDON_GEMM_VALUE, POSEIDON_GEMM_VALUE)};
#undef POSEIDON_GEMM_VALUE
    assert(!is_contained(v, nullptr) && "a dispatch argument is not set");
    return v;
  }
};
} // namespace

static Error readDispatchDescriptors(const Options &O, Module &M,
                                     std::map<std::string, LaunchDesc> &descs) {
  return readDescriptors(
      O, M, LaunchKind::HostDispatch, [&](ArrayRef<StringRef> toks) -> Error {
        DescriptorCursor Cursor(toks);
        LaunchDesc d;
        d.kernel = toks[0].str();
        d.cParam = Cursor.integer<int>("C argument");
        d.aParam = Cursor.integer<int>("A argument");
        d.bParam = Cursor.integer<int>("B argument");
        d.standalone = Cursor.integer<int>("standalone flag") != 0;
        d.modulusField = Cursor.integer<unsigned>("modulus field");
        if (d.modulusField > kOzakiIIMaxModuli)
          Cursor.fail("its modulus field " + Twine(d.modulusField) +
                      " is above " + Twine(kOzakiIIMaxModuli));
        d.aColMajor = Cursor.integer<int>("A layout") != 0;
        d.bColMajor = Cursor.integer<int>("B layout") != 0;
        d.cColMajor = Cursor.integer<int>("C layout") != 0;
        unsigned scheme = Cursor.integer<unsigned>("scheme");
        if (scheme > (unsigned)Scheme::Native)
          Cursor.fail("its scheme " + Twine(scheme) + " is above " +
                      Twine((unsigned)Scheme::Native));
        d.scheme = (Scheme)scheme;
        // Keyword blocks (see emitDescriptor), of which rd is required.
        bool haveDims = false;
        while (!Cursor.atEnd()) {
          if (Cursor.consume("rd")) {
            for (RuntimeDim &x : d.dims) {
              unsigned nf = Cursor.integer<unsigned>("factor count");
              if (nf > kMaxDimFactors)
                Cursor.fail("its factor count " + Twine(nf) + " is above " +
                            Twine(kMaxDimFactors));
              for (unsigned f = 0; f < nf && Cursor.ok(); ++f) {
                LaunchRef r;
                r.param = Cursor.integer<int>("factor argument");
                r.offset = Cursor.integer<int64_t>("factor offset");
                x.factors.push_back(r);
              }
              x.mul = Cursor.integer<int64_t>("dimension factor");
              x.add = Cursor.integer<int64_t>("dimension term");
              for (int64_t c : {x.mul, x.add})
                if (c < -kMaxDimConstant || c > kMaxDimConstant)
                  Cursor.fail("its dimension constant " + Twine(c) +
                              " is out of the int range");
              if (!x.isConst() && !d.standalone)
                Cursor.fail("it is fused and has a runtime dimension");
            }
            d.beta = Cursor.real("beta");
            haveDims = true;
          } else if (Cursor.consume("pf")) {
            d.cOffset = Cursor.integer<int64_t>("C offset");
            d.aOffset = Cursor.integer<int64_t>("A offset");
            d.bOffset = Cursor.integer<int64_t>("B offset");
          } else if (Cursor.consume("eq")) {
            unsigned ne = Cursor.integer<unsigned>("equality count");
            if (ne > kMaxAssumedEqual)
              Cursor.fail("its equality count " + Twine(ne) + " is above " +
                          Twine(kMaxAssumedEqual));
            for (unsigned e = 0; e < ne && Cursor.ok(); ++e) {
              LaunchRef x, y;
              x.param = Cursor.integer<int>("equality argument");
              x.offset = Cursor.integer<int64_t>("equality offset");
              y.param = Cursor.integer<int>("equality argument");
              y.offset = Cursor.integer<int64_t>("equality offset");
              d.assumedEqual.push_back({x, y});
            }
          } else if (Cursor.consume("lc")) {
            d.cropMAxis = Cursor.integer<int>("M axis");
            d.cropNAxis = Cursor.integer<int>("column axis");
            if (d.cropMAxis < 0 || d.cropMAxis > 2 || d.cropNAxis < 0 ||
                d.cropNAxis > 2 || d.cropMAxis == d.cropNAxis)
              Cursor.fail(
                  "its launch axes are not two distinct axes of x, y, z");
            d.launchCrop = true;
          } else {
            break;
          }
        }
        if (!haveDims)
          Cursor.fail("it has no rd block");
        if (Error E = Cursor.takeError())
          return E;
        descs[d.kernel] = d;
        return Error::success();
      });
}

// Kernel argument `r.param` lives where entry r.param of the launch's argument
// array points, as the CUDA runtime reads it; a field of an aggregate argument
// is at its offset from there in every host ABI, which the stub's own
// parameters are not (a small aggregate arrives split into registers). An
// integer is read as `T` at its address: recognition admits no integer
// narrower than 32 bits, and on the little-endian hosts the runtimes support
// the first 32 bits of a wider one are its low 32 bits.
static Value *loadLaunchValue(IRBuilder<> &B, Value *argArray,
                              const LaunchRef &r, Type *T) {
  Type *ptr = B.getPtrTy();
  Value *at = B.CreateLoad(
      ptr, B.CreateConstInBoundsGEP1_32(ptr, argArray, (unsigned)r.param));
  if (r.offset > 0)
    at = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), at, (uint64_t)r.offset);
  return B.CreateLoad(T, at);
}

// Builds the dispatch in front of the stub's launch, from the launch's own
// argument array. The launch is removed (standalone) or kept (fused); with
// assumed equalities it runs instead of the dispatch whenever one of them does
// not hold. A fused body's stub without that launch is an error: the device
// compile already fissioned its kernel, which computes nothing without the
// dispatch.
static Expected<bool> rewriteStub(const Options &O, Module &M, Function &F,
                                  const LaunchDesc &d) {
  CallInst *launch = findLaunchCall(F);
  CallInst *pop = findPopConfig(F);
  if (!d.standalone &&
      (!launch || (d.launchCrop && !pop) || !d.assumedEqual.empty()))
    return userError(F, F.getSubprogram(),
                     Twine("the stub ") + F.getName() +
                         " of a fused host-dispatched GEMM has no launch to "
                         "read the operands from, or a condition to check "
                         "first, and its kernel was fissioned device-side");
  if (!launch || (d.launchCrop && !pop)) {
    diagnose(F, F.getSubprogram(),
             "the library call of stub " + F.getName() +
                 " has no launch to read the kernel's arguments from; the "
                 "kernel is launched as written",
             DS_Warning);
    return false;
  }
  LLVMContext &C = M.getContext();
  Type *ptr = getInt8PtrTy(C);
  Type *i32 = Type::getInt32Ty(C);
  Type *i64 = Type::getInt64Ty(C);
  Type *dbl = Type::getDoubleTy(C);
  auto I32 = [&](int64_t v) { return ConstantInt::get(i32, (uint64_t)v); };
  auto I64 = [&](int64_t v) { return ConstantInt::get(i64, (uint64_t)v); };
  Value *argArray = launchArguments(*launch);
  // Resolved per descriptor: the scheme is a per-site solver decision.
  FunctionCallee callee = getDispatchCallee(M, d.scheme);

  IRBuilder<> B(launch);
  auto value = [&](const LaunchRef &r, Type *T) {
    return loadLaunchValue(B, argArray, r, T);
  };
  // Whether every runtime dimension fits the int the dispatch takes it as.
  // The dimensions are evaluated in i64 and every intermediate value is
  // checked: while all checks pass, each operand is below 2^31 and nothing
  // wraps.
  Value *inRange = nullptr;
  auto checked = [&](Value *v) {
    Value *fits = B.CreateICmpULE(v, I64(kMaxDimConstant));
    inRange = inRange ? B.CreateAnd(inRange, fits) : fits;
    return v;
  };
  auto dim = [&](const RuntimeDim &r) -> Value * {
    if (r.isConst())
      return I32(r.add);
    Value *v = nullptr;
    for (const LaunchRef &f : r.factors) {
      Value *x = checked(B.CreateSExt(value(f, i32), i64));
      v = v ? checked(B.CreateMul(v, x)) : x;
    }
    if (r.mul != 1)
      v = checked(B.CreateMul(v, I64(r.mul)));
    if (r.add)
      v = checked(B.CreateAdd(v, I64(r.add)));
    return B.CreateTrunc(v, i32);
  };
  auto cropDim = [&](int axis, const RuntimeDim &crop) -> Value * {
    Value *g = B.CreateLoad(
        i32, B.CreateConstInBoundsGEP1_32(i32, poppedGrid(*pop), axis));
    Value *b = B.CreateLoad(
        i32, B.CreateConstInBoundsGEP1_32(i32, poppedBlock(*pop), axis));
    Value *g64 = B.CreateZExt(g, i64);
    Value *b64 = B.CreateZExt(b, i64);
    Value *ext = B.CreateMul(g64, b64);
    Value *below = B.CreateICmpULT(ext, I64(crop.add));
    return B.CreateTrunc(B.CreateSelect(below, ext, I64(crop.add)), i32);
  };

  Value *Cp = value({d.cParam, d.cOffset}, ptr);
  Value *Ap = value({d.aParam, d.aOffset}, ptr);
  Value *Bp = value({d.bParam, d.bOffset}, ptr);
  DispatchArgs a;
  a.C = Cp;
  a.A = Ap;
  a.B = Bp;
  a.M = d.launchCrop ? cropDim(d.cropMAxis, d.dims[DimM]) : dim(d.dims[DimM]);
  a.Ncols = d.launchCrop ? cropDim(d.cropNAxis, d.dims[DimNcols])
                         : dim(d.dims[DimNcols]);
  a.K = dim(d.dims[DimK]);
  a.lda = dim(d.dims[DimLda]);
  a.ldb = dim(d.dims[DimLdb]);
  a.ldc = dim(d.dims[DimLdc]);
  a.aColMajor = I32(d.aColMajor ? 1 : 0);
  a.bColMajor = I32(d.bColMajor ? 1 : 0);
  a.cColMajor = I32(d.cColMajor ? 1 : 0);
  a.alpha = ConstantFP::get(dbl, 1.0);
  a.beta = ConstantFP::get(dbl, d.beta);
  a.stream = launchStreamHandle(B, *launch);
  SmallVector<Value *, 16> args = a.values();
  if (dispatchHasMode(d.scheme))
    args.push_back(I32(d.modulusField));
  Value *holds = nullptr;
  for (const auto &[x, y] : d.assumedEqual) {
    Value *lhs = value(x, i32);
    Value *rhs = value(y, i32);
    Value *eq = B.CreateICmpEQ(lhs, rhs);
    holds = holds ? B.CreateAnd(holds, eq) : eq;
  }
  assert((d.standalone || !inRange) &&
         "the reader admits no runtime dimension in a fused descriptor");
  if (inRange)
    holds = holds ? B.CreateAnd(holds, inRange) : inRange;

  Value *success = Constant::getNullValue(launch->getType());
  if (!holds) {
    B.CreateCall(callee, args);
    if (d.standalone) {
      launch->replaceAllUsesWith(success);
      launch->eraseFromParent();
    }
  } else {
    Instruction *thenTerm = nullptr, *elseTerm = nullptr;
    SplitBlockAndInsertIfThenElse(holds, launch->getIterator(), &thenTerm,
                                  &elseTerm);
    IRBuilder<>(thenTerm).CreateCall(callee, args);
    BasicBlock *join = launch->getParent();
    launch->moveBefore(elseTerm->getIterator());
    if (!launch->use_empty()) {
      PHINode *phi = PHINode::Create(launch->getType(), 2, "", join->begin());
      launch->replaceAllUsesWith(phi);
      phi->addIncoming(success, thenTerm->getParent());
      phi->addIncoming(launch, elseTerm->getParent());
    }
  }
  if (!O.Diagnostics.Print)
    return true;
  errs() << "[poseidon] " << (d.standalone ? "replaced" : "prepended")
         << " the launch in stub " << F.getName() << " with "
         << dispatchSym(d.scheme)
         << " (C=" << launchRefString({d.cParam, d.cOffset}, "arg")
         << " A=" << launchRefString({d.aParam, d.aOffset}, "arg")
         << " B=" << launchRefString({d.bParam, d.bOffset}, "arg") << ')';
  for (const auto &[x, y] : d.assumedEqual)
    errs() << ", if " << launchRefString(x, "arg")
           << " == " << launchRefString(y, "arg");
  errs() << '\n';
  return true;
}

// Runs before inlining (PipelineStart), while `__device_stub__K` is still a
// function of its own that ends in the kernel's launch.
Expected<bool> rewriteGemmStubBodies(const Options &O, Module &M) {
  std::map<std::string, LaunchDesc> descs;
  if (Error E = readDispatchDescriptors(O, M, descs))
    return std::move(E);

  bool changed = false;
  for (Function &F : M) {
    if (F.isDeclaration() || !F.getReturnType()->isVoidTy())
      continue;
    auto dIt = llvm::find_if(
        descs, [&](const auto &kv) { return isLaunchStubFor(F, kv.first); });
    if (dIt == descs.end())
      continue;
    Expected<bool> rewritten = rewriteStub(O, M, F, dIt->second);
    if (!rewritten)
      return rewritten.takeError();
    changed |= *rewritten;
  }
  return changed;
}

} // namespace poseidon
