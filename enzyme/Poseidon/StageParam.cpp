//===- StageParam.cpp - df64 staging across a parameter array ------------===//
//
// See StageParam.h.
//===---------------------------------------------------------------------===//
#include "StageParam.h"
#include "Flags.h"
#include "HostDispatch.h"
#include "LaunchDescriptors.h"
#include "Optimize.h"

#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>
#include <vector>

using namespace llvm;

namespace poseidon {

namespace {

// One emitF64ToDS split of a load:
//   hiT = fptrunc(L) ; los = fsub double(L, hib=fpext(hiT)) ; loT =
//   fptrunc(los)
struct DekkerSplit {
  FPTruncInst *hiT = nullptr;
  BinaryOperator *los = nullptr;
  FPExtInst *hib = nullptr;
  FPTruncInst *loT = nullptr;
};

// Collect every Dekker split hanging off `L`, plus the bare fptruncs that are
// the `hi` of no split. One load can feed several units, each emitting its own
// split before CSE, which matchDekkerSplit in Optimize.cpp refuses. Every other
// reader is served at the rewrite by the exact reconstruction hi + lo. Returns
// false only for one `hi` fptrunc feeding two residual subtractions, whose
// erase order would be ambiguous.
static bool collectDekkerSplits(LoadInst *L, SmallVectorImpl<DekkerSplit> &out,
                                SmallVectorImpl<FPTruncInst *> &bareTruncs) {
  SmallPtrSet<FPTruncInst *, 4> pairedHi;
  for (User *U : L->users()) {
    auto *los = dyn_cast<BinaryOperator>(U);
    if (!los || los->getOpcode() != Instruction::FSub ||
        !los->getType()->isDoubleTy() || los->getOperand(0) != L ||
        !los->hasOneUse())
      continue;
    auto *hib = dyn_cast<FPExtInst>(los->getOperand(1));
    if (!hib || !hib->hasOneUse())
      continue;
    auto *hiT = dyn_cast<FPTruncInst>(hib->getOperand(0));
    if (!hiT || hiT->getOperand(0) != L || !hiT->getType()->isFloatTy())
      continue;
    auto *loT = dyn_cast<FPTruncInst>(*los->user_begin());
    if (!loT || !loT->getType()->isFloatTy())
      continue;
    if (!pairedHi.insert(hiT).second)
      return false;
    out.push_back({hiT, los, hib, loT});
  }
  for (User *U : L->users())
    if (auto *FT = dyn_cast<FPTruncInst>(U))
      if (FT->getType()->isFloatTy() && !pairedHi.contains(FT))
        bareTruncs.push_back(FT);
  return true;
}

struct StagedParam {
  unsigned argIdx = 0;
  SmallVector<LoadInst *, 64> loads;
  unsigned splitLoads = 0; // loads carrying at least one Dekker split
  unsigned reconLoads = 0; // loads whose readers get the hi + lo restore
};

// Strip GEP / bitcast / addrspacecast back to the Argument `v` roots in.
static Argument *traceToArgument(Value *v) {
  while (v) {
    if (auto *A = dyn_cast<Argument>(v))
      return A;
    if (auto *GEP = dyn_cast<GEPOperator>(v)) {
      v = GEP->getPointerOperand();
      continue;
    }
    if (auto *CI = dyn_cast<CastInst>(v)) {
      v = CI->getOperand(0);
      continue;
    }
    if (auto *CE = dyn_cast<ConstantExpr>(v)) {
      if (CE->getOpcode() == Instruction::BitCast ||
          CE->getOpcode() == Instruction::AddrSpaceCast ||
          CE->getOpcode() == Instruction::GetElementPtr) {
        v = CE->getOperand(0);
        continue;
      }
    }
    return nullptr;
  }
  return nullptr;
}

// Walk the pointer graph rooted at F's argument `ai`: every derived pointer
// must be a GEP and every access a simple `load double`, and at least one load
// must carry a Dekker split.
static bool collectStagedParam(Function &F, Function *Proxy, unsigned ai,
                               StagedParam &out) {
  Type *DoubleTy = Type::getDoubleTy(F.getContext());
  Argument *A = F.getArg(ai);
  if (!A->getType()->isPointerTy())
    return false;

  // Checked before any IR changes: the host will replace the wrapper's launch
  // argument with a limb buffer, so this parameter must reach such an argument
  // and nothing else in the wrapper may read it.
  Value *SA = siteArg(Proxy, ai);
  Argument *WA = SA ? traceToArgument(SA) : nullptr;
  if (!WA) {
    if (flags::Print)
      errs() << "[stage-param] arg" << ai
             << " bail: does not root in a launch argument of the wrapper "
                "kernel\n";
    return false;
  }
  if (!WA->hasOneUse() || !isa<CallBase>(*WA->user_begin())) {
    if (flags::Print)
      errs() << "[stage-param] arg" << ai
             << " bail: the wrapper's launch argument is read outside the "
                "annotated call, so it cannot be replaced by a limb buffer\n";
    return false;
  }

  SmallPtrSet<Value *, 32> seen;
  SmallVector<Value *, 16> work;
  work.push_back(A);
  seen.insert(A);
  while (!work.empty()) {
    Value *P = work.pop_back_val();
    for (User *U : P->users()) {
      auto *I = dyn_cast<Instruction>(U);
      if (!I || I->getFunction() != &F) {
        if (flags::Print)
          errs() << "[stage-param] arg" << ai
                 << " bail: non-instruction or out-of-function user " << *U
                 << "\n";
        return false;
      }
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        // The layout is preserved, so the GEP is only walked through; a GEP
        // whose base is not P means the pointer escaped into address
        // arithmetic not modelled here.
        if (GEP->getPointerOperand() != P)
          return false;
        if (seen.insert(GEP).second)
          work.push_back(GEP);
      } else if (auto *L = dyn_cast<LoadInst>(U)) {
        if (L->getType() != DoubleTy || L->getPointerOperand() != P ||
            !L->isSimple()) {
          if (flags::Print)
            errs() << "[stage-param] arg" << ai << " bail: load " << *L << "\n";
          return false;
        }
        SmallVector<DekkerSplit, 4> splits;
        SmallVector<FPTruncInst *, 4> bare;
        if (!collectDekkerSplits(L, splits, bare)) {
          if (flags::Print)
            errs() << "[stage-param] arg" << ai
                   << " bail: ambiguous split shape on " << *L << "\n";
          return false;
        }
        // A loaded value that is stored somewhere else is a staging copy owned
        // by the shared-buffer arms; hoisting the split past that boundary
        // would reconstruct an F64 only to split it again. Checked on the load
        // and on each limb, since the shared-buffer arm may already have turned
        // the store into a {hi,lo} pair.
        auto storedSomewhere = [](Value *V) {
          for (User *U : V->users())
            if (isa<StoreInst>(U))
              return true;
          return false;
        };
        bool stagingCopy = storedSomewhere(L);
        for (DekkerSplit &sp : splits)
          stagingCopy |= storedSomewhere(sp.hiT) || storedSomewhere(sp.loT);
        for (FPTruncInst *FT : bare)
          stagingCopy |= storedSomewhere(FT);
        if (stagingCopy) {
          if (flags::Print)
            errs() << "[stage-param] arg" << ai
                   << " bail: the load is a staging copy into another buffer "
                      "(the shared-staging arm owns it): "
                   << *L << "\n";
          return false;
        }
        if (splits.empty())
          ++out.reconLoads;
        else
          ++out.splitLoads;
        out.loads.push_back(L);
      } else {
        if (flags::Print)
          errs() << "[stage-param] arg" << ai << " bail: unexpected user " << *U
                 << "\n";
        return false;
      }
    }
  }
  out.argIdx = ai;
  return out.splitLoads > 0;
}

} // namespace

unsigned stageParamArrayDS(Function &F, Function *Proxy,
                           SmallVectorImpl<unsigned> *stagedParams) {
  if (F.isDeclaration())
    return 0;
  if (!Proxy)
    Proxy = &F;
  // The joint solver materializes after optimizeSiteBody has written the
  // descriptor, so a joint solve would rewrite the kernel to read limbs and
  // never tell the host to produce them; refuse the whole arm.
  if (flags::JointDP) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      errs() << "[poseidon] -poseidon-stage-param-arrays is not wired for "
                "-poseidon-joint-dp (the joint materialize runs after the "
                "descriptor is written); parameter-array staging is off for "
                "this compilation\n";
    }
    return 0;
  }
  LLVMContext &Ctx = F.getContext();
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *I8Ty = Type::getInt8Ty(Ctx);

  unsigned staged = 0;
  for (unsigned ai = 0; ai < F.arg_size(); ++ai) {
    StagedParam sp;
    if (!collectStagedParam(F, Proxy, ai, sp))
      continue;

    for (LoadInst *L : sp.loads) {
      SmallVector<DekkerSplit, 4> splits;
      SmallVector<FPTruncInst *, 4> bare;
      bool ok = collectDekkerSplits(L, splits, bare);
      (void)ok;
      assert(ok && "recognition and rewrite disagree on a staged load");
      IRBuilder<> B(L);
      Value *p = L->getPointerOperand();
      // hi keeps the slot's own alignment; lo sits 4 bytes in.
      auto *nhi = B.CreateAlignedLoad(FloatTy, p, L->getAlign(), "ds.stg.hi");
      auto *nlo = B.CreateAlignedLoad(
          FloatTy, B.CreateGEP(I8Ty, p, B.getInt64(4)), Align(4), "ds.stg.lo");
      // The limb buffer is scratch this pass owns, filled before the launch and
      // never written through, so it is invariant for the whole kernel (the
      // guarantee `const __restrict__` gives a hand-written kernel), which lets
      // NVPTX issue the read-only-cache load.
      MDNode *inv = MDNode::get(Ctx, {});
      nhi->setMetadata(LLVMContext::MD_invariant_load, inv);
      nlo->setMetadata(LLVMContext::MD_invariant_load, inv);
      for (DekkerSplit &s : splits) {
        s.hiT->replaceAllUsesWith(nhi);
        s.loT->replaceAllUsesWith(nlo);
        s.loT->eraseFromParent();
        s.los->eraseFromParent();
        s.hib->eraseFromParent();
        s.hiT->eraseFromParent();
      }
      for (FPTruncInst *FT : bare) {
        FT->replaceAllUsesWith(nhi);
        FT->eraseFromParent();
      }
      // Every remaining reader (an fcmp guard, a PHI out of an index-guarded
      // block, a unit still at FP64) gets hi + lo, which is exact: the pair is
      // that double's own Dekker split.
      if (!L->use_empty()) {
        auto *rec = cast<Instruction>(B.CreateFAdd(B.CreateFPExt(nhi, DoubleTy),
                                                   B.CreateFPExt(nlo, DoubleTy),
                                                   "ds.stg.f64"));
        // Same tag emitDSToF64 sets: the pair is normalized and hi + lo
        // exact, so foldDSPairRoundtrip can cancel a re-split.
        rec->setMetadata("poseidon.ds.join", MDNode::get(Ctx, {}));
        L->replaceAllUsesWith(rec);
      }
      L->eraseFromParent();
    }

    ++staged;
    if (stagedParams)
      stagedParams->push_back(ai);
    errs() << "[poseidon] staged parameter array arg" << ai << " of "
           << F.getName() << " as df64 limbs (" << sp.loads.size() << " loads, "
           << sp.splitLoads << " Dekker-split, " << sp.reconLoads
           << " restored)\n";
  }
  return staged;
}

static NoteMap<StagedParamNote> &stagedNoteMap() {
  static NoteMap<StagedParamNote> m;
  return m;
}

void noteStagedBody(const Function *body, const StagedParamNote &n) {
  stagedNoteMap().note(body, n);
}

bool getStagedBody(const Function *body, StagedParamNote &out) {
  return stagedNoteMap().get(body, out);
}

void writeStageDescriptor(Function &wrapper, ArrayRef<Value *> primalArgs,
                          const CallInst *site, const StagedParamNote &n,
                          StringRef cacheDir) {
  SmallVector<int, 4> launchArgs;
  for (unsigned bp : n.bodyParams) {
    // Recognition established both of these before touching the IR; failing
    // here would leave a kernel that reads limbs with no host side to produce
    // them, so it is a hard error.
    int a = bp < primalArgs.size() ? traceToArgIndex(primalArgs[bp]) : -1;
    if (a < 0)
      report_fatal_error("stage-param: a staged body parameter does not map to "
                         "a launch argument of its wrapper kernel, but the "
                         "kernel has already been rewritten to read limbs");
    for (const User *U : wrapper.getArg(a)->users())
      if (U != site)
        report_fatal_error("stage-param: the wrapper kernel's launch argument "
                           "is read outside the annotated call, but the kernel "
                           "has already been rewritten to read limbs");
    launchArgs.push_back(a);
  }

  if (launchArgs.empty()) {
    // Nothing staged this time: remove any descriptor an earlier solve left, so
    // the host can never stage a kernel that reads doubles.
    removeDescriptor(cacheDir, wrapper.getName(), kStageScheme);
    return;
  }

  if (!writeDescriptor(cacheDir, wrapper.getName(), kStageScheme,
                       "[stage-param]", [&](raw_ostream &os) {
                         for (int a : launchArgs)
                           os << ' ' << a;
                       }))
    return;
  errs() << "[stage-param] wrote descriptor for " << wrapper.getName()
         << ": limb-staged launch arg(s)";
  for (int a : launchArgs)
    errs() << ' ' << a;
  errs() << "\n";
}

namespace {
struct StageDesc {
  std::string kernel;
  std::vector<int> args;
};
} // namespace

static bool readStageDescriptors(StringRef cacheDir,
                                 std::vector<StageDesc> &out) {
  readDescriptors(cacheDir, kStageScheme, [&](ArrayRef<StringRef> toks) {
    if (toks.size() < 2)
      return;
    StageDesc d;
    d.kernel = toks[0].str();
    for (unsigned i = 1; i < toks.size(); ++i) {
      int a = -1;
      if (toks[i].getAsInteger(10, a) || a < 0) {
        d.args.clear();
        break;
      }
      d.args.push_back(a);
    }
    if (!d.args.empty())
      out.push_back(std::move(d));
  });
  return !out.empty();
}

// void *__poseidon_stage_split_f64(const void *src, void *stream)
static FunctionCallee getStageSplit(Module &M) {
  LLVMContext &C = M.getContext();
  Type *ptr = PointerType::get(C, 0);
  FunctionType *FT = FunctionType::get(ptr, {ptr, ptr}, false);
  return M.getOrInsertFunction("__poseidon_stage_split_f64", FT);
}

bool rewriteStageStubBodies(Module &M, StringRef cacheDir) {
  std::vector<StageDesc> descs;
  if (!readStageDescriptors(cacheDir, descs))
    return false;

  LLVMContext &C = M.getContext();
  Type *ptr = PointerType::get(C, 0);
  bool changed = false;

  for (Function &F : M) {
    if (F.isDeclaration() || !F.getReturnType()->isVoidTy())
      continue;
    const StageDesc *dp = nullptr;
    for (const StageDesc &d : descs) {
      if (isLaunchStubFor(F, d.kernel)) {
        dp = &d;
        break;
      }
    }
    if (!dp)
      continue;
    // PipelineStart can run more than once over the host module; a second
    // split call would redo the same O(N) work into the same buffer.
    if (F.hasFnAttribute("poseidon-df64-limb-staged"))
      continue;

    CallInst *launch = nullptr, *pop = nullptr;
    for (Instruction &I : instructions(F))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (const Function *cal = CI->getCalledFunction()) {
          if (cal->getName() == "cudaLaunchKernel" && !launch)
            launch = CI;
          else if (cal->getName() == "__cudaPopCallConfiguration" && !pop)
            pop = CI;
        }
    if (!launch || !pop || pop->arg_size() < 4) {
      errs() << "[stage-param] " << F.getName()
             << ": no cudaLaunchKernel / __cudaPopCallConfiguration pair in "
                "the launch stub; refusing to stage (the device kernel reads "
                "limbs, so emitting nothing here would be wrong)\n";
      report_fatal_error("stage-param: unrecognized CUDA launch stub");
    }

    IRBuilder<> B(launch);
    Value *stream = B.CreateLoad(ptr, pop->getArgOperand(3), "stgp.stream");
    for (int a : dp->args) {
      if ((unsigned)a >= F.arg_size() || !F.getArg(a)->getType()->isPointerTy())
        report_fatal_error("stage-param: descriptor names a launch argument "
                           "the stub does not have");
      Argument *Arg = F.getArg(a);
      // The stub stores each launch argument into a home alloca and passes that
      // alloca's address in the cudaLaunchKernel argument array, so re-storing
      // the limb pointer into the same slot substitutes it for this launch.
      StoreInst *home = nullptr;
      unsigned homes = 0;
      for (User *U : Arg->users())
        if (auto *S = dyn_cast<StoreInst>(U))
          if (S->getValueOperand() == Arg &&
              isa<AllocaInst>(S->getPointerOperand())) {
            home = S;
            ++homes;
          }
      if (homes != 1)
        report_fatal_error("stage-param: could not find the launch argument's "
                           "home slot in the launch stub");
      Value *limb = B.CreateCall(getStageSplit(M), {Arg, stream}, "stgp.limb");
      B.CreateStore(limb, home->getPointerOperand());
    }
    F.addFnAttr("poseidon-df64-limb-staged");
    errs() << "[stage-param] prepended the df64 split to " << F.getName()
           << " (launch arg(s)";
    for (int a : dp->args)
      errs() << ' ' << a;
    errs() << ")\n";
    changed = true;
  }
  return changed;
}

} // namespace poseidon
