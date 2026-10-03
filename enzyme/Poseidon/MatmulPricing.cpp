// Matmul candidate pricing: option labels, per-tile MMA cost, measured
// in-kernel rows, the WMMA target table and shared-memory capacity.
#include "CostModel.h"
#include "Evaluators.h"
#include "Flags.h"
#include "InKernelRaise.h"
#include "MatmulInternal.h"
#include "Optimize.h"
#include "Solvers.h"
#include "Utils.h"
#include "WmmaUtils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <set>
#include <string>

using namespace llvm;

namespace poseidon {

std::string mmaShapeSuffix(unsigned M, unsigned N, unsigned K) {
  return ("m" + Twine(M) + "n" + Twine(N) + "k" + Twine(K)).str();
}

std::string matmulOptionLabel(const CandidateMatmul::Option &opt) {
  switch (opt.strategy) {
  case CandidateMatmul::Option::Strategy::Direct:
    return "wmma " + mmaShapeSuffix(opt.tileM, opt.tileN, opt.tileK) + " " +
           std::string(fpKindName(opt.inputPrec)) + "/" +
           fpKindName(opt.accPrec);
  case CandidateMatmul::Option::Strategy::OzakiI:
    return ozakiIOptionLabel(opt);
  case CandidateMatmul::Option::Strategy::OzakiII:
    // strategyParam encodes num_moduli for OzakiII (same role as N for OzakiI).
    // nm=0 is the native cuBLAS DGEMM dispatch.
    if (opt.strategyParam == 0)
      return "native-dgemm cublas f64/f64";
    return (Twine("ozaki-ii nm=") + Twine(opt.strategyParam) + " wmma m" +
            Twine(opt.tileM) + "n" + Twine(opt.tileN) + "k" + Twine(opt.tileK) +
            " s8/s32")
        .str();
  case CandidateMatmul::Option::Strategy::TcecDispatch:
    return "tcec-dispatch wmma m" + std::to_string(opt.tileM) + "n" +
           std::to_string(opt.tileN) + "k" + std::to_string(opt.tileK) + " " +
           fpKindName(opt.inputPrec) + "/" + fpKindName(opt.accPrec);
  case CandidateMatmul::Option::Strategy::DirectDispatch:
    return "direct-dispatch cublas " + std::string(fpKindName(opt.inputPrec)) +
           "/" + fpKindName(opt.accPrec);
  }
  llvm_unreachable("unhandled CandidateMatmul::Option::Strategy");
}

// TODO: include more wmma_load_{a,b,c} and wmma_store_d costs
// Today only m16n16k16 has them.
// Returns the per-tile-MMA reciprocal-throughput cost (cycles per tile op).
double getMatmulCompCost(unsigned M, unsigned N, unsigned K, FPKind inputPrec,
                         FPKind accPrec) {
  if (inputPrec == FPKind::Invalid || accPrec == FPKind::Invalid)
    report_fatal_error("getMatmulCompCost: unexpected FPKind::Invalid");
  std::string opcode = "wmma_mma_" + mmaShapeSuffix(M, N, K);
  std::string prec =
      std::string(fpKindName(inputPrec)) + "_" + fpKindName(accPrec);
  const auto &model = getCostModel();
  auto it = model.find({opcode, prec});
  if (it == model.end())
    report_fatal_error("cost model missing row for " + Twine(opcode) + " " +
                       Twine(prec));
  return it->second;
}

// Optional measured pricing for in-kernel candidates: one row per class, in the
// same unit and protocol as ozaki_dispatch_rel (candidate wall clock / scalar
// FP64 baseline wall clock),
//     wmma_inkernel_rel,<class>,<rel>
// with <class> naming the candidate as the solver's table does ("f16_f32",
// "tcec_n2_f16_f32"). Composing a price from per-MMA rows omits the cooperative
// fill, the shared-memory staging and the accumulator traffic.
std::string inKernelDirectClass(FPKind inputPrec, FPKind accPrec) {
  return std::string(fpKindName(inputPrec)) + "_" + fpKindName(accPrec);
}
std::string inKernelTcecClass(unsigned n, FPKind inputPrec, FPKind accPrec) {
  return "tcec_n" + std::to_string(n) + "_" + fpKindName(inputPrec) + "_" +
         fpKindName(accPrec);
}
// Measured rel for a class, or -1.0 when uncalibrated. The rel is shape
// dependent (fill and padding scale with the tile), so a shape-qualified row
//     wmma_inkernel_rel,<class>_m<M>n<N>k<K>,<rel>
// is preferred and the unqualified wmma_inkernel_rel,<class>,<rel> is the
// fallback.
static double lookupInKernelRel(const std::string &cls, unsigned tileM,
                                unsigned tileN, unsigned tileK) {
  double q =
      queryCostModelOr("wmma_inkernel_rel",
                       cls + "_" + mmaShapeSuffix(tileM, tileN, tileK), -1.0);
  if (q > 0.0)
    return q;
  return queryCostModelOr("wmma_inkernel_rel", cls, -1.0);
}

// Price one in-kernel raise from its measured wmma_inkernel_rel row, or refuse
// to propose it. There is deliberately no composed fallback: per-MMA rows model
// only the mma chain and measure 3.5-64x optimistic against the realized rate,
// and Poseidon does not substitute a model for a missing measurement anywhere
// else.
bool priceInKernelRaiseFromMeasuredRow(const AbstractMatmul &m,
                                       const std::string &cls, unsigned tileM,
                                       unsigned tileN, unsigned tileK,
                                       double baselinePerMac,
                                       const std::string &what,
                                       double &costOut) {
  double rel = lookupInKernelRel(cls, tileM, tileN, tileK);
  if (rel > 0.0) {
    costOut = baselinePerMac * rel;
    return true;
  }

  const std::string key = cls + "_" + mmaShapeSuffix(tileM, tileN, tileK);
  if (flags::InKernelCalibration) {
    if (flags::ApplyRewrites.empty())
      report_fatal_error(
          "Poseidon: -poseidon-inkernel-calibration requires "
          "-poseidon-apply-rewrites. Uncalibrated in-kernel classes are "
          "proposed "
          "at a placeholder price under that flag, and a DP solve must never "
          "see one.");
    costOut = baselinePerMac;
    llvm::errs() << "[poseidon] CALIBRATION MODE: " << what
                 << " has no measured wmma_inkernel_rel," << key
                 << " row; proposing it at a PLACEHOLDER price (rel=1) so it "
                    "can be materialized and timed. Costs in this build are "
                    "not decisions.\n";
    return true;
  }

  static std::set<std::string> reported;
  if (reported.insert(key).second) {
    llvm::errs()
        << "[poseidon] WARNING: matmul[" << m.id << "] " << m.M << "x" << m.N
        << "x" << m.K << ": " << what
        << " NOT PROPOSED -- this device's cost model carries no measured "
           "wmma_inkernel_rel,"
        << key << " (or wmma_inkernel_rel," << cls
        << ") row. Composing the price from per-MMA rows would model the mma "
           "chain alone (no cooperative fill, no shared-memory staging, no "
           "accumulator traffic, no software pipelining) and measures 3.5-64x "
           "optimistic across the classes timed so far, so the class is "
           "refused rather than priced on an unmeasured model. Run "
           "poseidon-calibrate on this device.\n";
  }
  if (flags::StrictMode)
    report_fatal_error(
        "Poseidon strict mode: in-kernel raise class '" + Twine(key) +
        "' has no measured wmma_inkernel_rel row in " + Twine(costModelPath()));
  return false;
}

static constexpr WmmaTarget kExpectedWmmaTargets[] = {
    {16, 16, 16, FPKind::F16, FPKind::F16},
    {16, 16, 16, FPKind::F16, FPKind::F32},
    {16, 16, 16, FPKind::BF16, FPKind::F32},
    {32, 8, 16, FPKind::F16, FPKind::F16},
    {32, 8, 16, FPKind::F16, FPKind::F32},
    {32, 8, 16, FPKind::BF16, FPKind::F32},
    {8, 32, 16, FPKind::F16, FPKind::F16},
    {8, 32, 16, FPKind::F16, FPKind::F32},
    {8, 32, 16, FPKind::BF16, FPKind::F32},
    {16, 16, 8, FPKind::TF32, FPKind::F32},
    {8, 8, 4, FPKind::F64, FPKind::F64},
    // INT8 tensor-core: m16n16k16 s8 input x s32 accumulator (IMMA)
    {16, 16, 16, FPKind::S8, FPKind::S32},
};

const SmallVector<WmmaTarget, 16> &getAvailableWmmaTargets() {
  static const SmallVector<WmmaTarget, 16> available = []() {
    SmallVector<WmmaTarget, 16> result;
    const auto &model = getCostModel();
    for (const WmmaTarget &t : kExpectedWmmaTargets) {
      std::string opcode = "wmma_mma_" + mmaShapeSuffix(t.M, t.N, t.K);
      std::string prec =
          std::string(fpKindName(t.inputPrec)) + "_" + fpKindName(t.accPrec);
      if (model.find({opcode, prec}) != model.end()) {
        result.push_back(t);
      } else {
        llvm::errs() << "Poseidon: cost model missing row for known wmma "
                        "intrinsic "
                     << opcode << " " << prec
                     << " — this target will not be considered.\n";
      }
    }
    return result;
  }();
  return available;
}

// Static shared-memory preconditions at proposal time (see
// kStaticShmemCap): a shape whose scratch exceeds the cap is
// unbuildable, and ptxas reports it as a whole-module failure.
uint64_t enclosingKernelSharedBytes(Function &F) {
  Module *mod = F.getParent();
  const DataLayout &DL = mod->getDataLayout();
  // Only the addrspace(3) globals co-resident with this raise count, so take
  // the max over the enclosing kernels rather than a module-wide sum (a
  // calibration harness holding a gold kernel next to the raised one would
  // otherwise be charged twice).
  auto sharedBytesUsedBy = [&](Function *root,
                               SmallPtrSetImpl<GlobalVariable *> &out) {
    SmallVector<Function *, 8> work{root};
    SmallPtrSet<Function *, 8> seen{root};
    while (!work.empty()) {
      Function *cur = work.pop_back_val();
      for (Instruction &I : instructions(*cur)) {
        for (Value *Op : I.operands()) {
          Value *base = Op->stripPointerCastsAndAliases();
          if (auto *GV = dyn_cast<GlobalVariable>(base))
            if (GV->getAddressSpace() == 3 && GV->getValueType()->isSized())
              out.insert(GV);
        }
        if (auto *CB = dyn_cast<CallBase>(&I))
          if (Function *Callee = CB->getCalledFunction())
            if (!Callee->isDeclaration() && seen.insert(Callee).second)
              work.push_back(Callee);
      }
    }
  };
  auto sizeOfSet = [&](const SmallPtrSetImpl<GlobalVariable *> &s) {
    uint64_t b = 0;
    for (GlobalVariable *GV : s)
      b += DL.getTypeAllocSize(GV->getValueType()).getFixedValue();
    return b;
  };
  uint64_t bytes = 0;
  SmallPtrSet<GlobalVariable *, 16> own;
  sharedBytesUsedBy(&F, own);
  std::string orig = F.getName().str();
  for (StringRef pfx : {"preprocess_", "fakeaug_", "augmented_", "diffe"})
    if (StringRef(orig).starts_with(pfx)) {
      orig = orig.substr(pfx.size());
      break;
    }
  Function *origFn = mod->getFunction(orig);
  bool sawKernel = false;
  if (origFn)
    for (User *U : origFn->users())
      if (auto *CB = dyn_cast<CallBase>(U))
        if (Function *K = CB->getFunction())
          if (K->getCallingConv() == CallingConv::PTX_Kernel) {
            SmallPtrSet<GlobalVariable *, 16> s(own.begin(), own.end());
            sharedBytesUsedBy(K, s);
            bytes = std::max(bytes, sizeOfSet(s));
            sawKernel = true;
          }
  if (!sawKernel) {
    // Module-wide sum, the conservative direction, even though a tighter answer
    // is available when F is itself the kernel.
    for (GlobalVariable &G : mod->globals())
      if (G.getAddressSpace() == 3 && G.getValueType()->isSized())
        bytes += DL.getTypeAllocSize(G.getValueType()).getFixedValue();
  }
  return bytes;
}

uint64_t inKernelRaiseSharedBytes(Module &M, const CandidateMatmul::Option &opt,
                                  std::string *breakdown) {
  const DataLayout &DL = M.getDataLayout();
  LLVMContext &ctx = M.getContext();
  auto bytesOf = [&](FPKind k) -> uint64_t {
    return DL.getTypeAllocSize(llvmTypeForFPKind(ctx, k)).getFixedValue();
  };
  const uint64_t mPad = (uint64_t)opt.mChain * opt.tileM;
  const uint64_t nPad = (uint64_t)opt.nChain * opt.tileN;
  const uint64_t tileK = opt.tileK;

  switch (opt.strategy) {
  case CandidateMatmul::Option::Strategy::Direct: {
    // Mirrors the Direct materializer's allocations: the D, A and B staging
    // tiles always exist.
    uint64_t in = bytesOf(opt.inputPrec), acc = bytesOf(opt.accPrec);
    uint64_t d = mPad * nPad * acc, a = mPad * tileK * in,
             b = tileK * nPad * in;
    if (breakdown)
      *breakdown = (Twine(d) + " B D tile (" + Twine(mPad) + "x" + Twine(nPad) +
                    " " + fpKindName(opt.accPrec) + ") + " + Twine(a) +
                    " B A tile + " + Twine(b) + " B B tile")
                       .str();
    return d + a + b;
  }
  case CandidateMatmul::Option::Strategy::OzakiI: {
    // Mirrors the Ozaki-I materializer: N slice buffers per side plus N F32
    // output tiles, all live at once.
    uint64_t slice =
        bytesOf(opt.inputPrec); // sliceScratchTy: f16/bf16 2, tf32 4
    uint64_t n = opt.strategyParam;
    uint64_t per =
        mPad * tileK * slice + tileK * nPad * slice + mPad * nPad * 4;
    if (breakdown)
      *breakdown =
          (Twine(n) + " slices x " + Twine(per) + " B (A " +
           Twine(mPad * tileK * slice) + " + B " + Twine(tileK * nPad * slice) +
           " + D f32 " + Twine(mPad * nPad * 4) + ", " + Twine(mPad) + "x" +
           Twine(nPad) + " output tile)")
              .str();
    return n * per;
  }
  case CandidateMatmul::Option::Strategy::OzakiII:
  case CandidateMatmul::Option::Strategy::TcecDispatch:
  case CandidateMatmul::Option::Strategy::DirectDispatch:
    return 0; // library call
  }
  llvm_unreachable("unhandled strategy in inKernelRaiseSharedBytes");
}

// Shared refusal for every in-kernel class. Returns true when the raise fits.
bool inKernelRaiseFits(Function *F, const AbstractMatmul &m,
                       const CandidateMatmul::Option &opt,
                       uint64_t existingBytes, StringRef className) {
  if (!F)
    return true;
  std::string breakdown;
  uint64_t raiseBytes =
      inKernelRaiseSharedBytes(*F->getParent(), opt, &breakdown);
  if (raiseBytes == 0)
    return true;
  uint64_t total = existingBytes + raiseBytes;
  if (total <= kStaticShmemCap)
    return true;
  if (flags::Print)
    llvm::errs()
        << "  Matmul[" << m.id << "] " << m.M << "x" << m.N << "x" << m.K
        << ": " << className << " NOT PROPOSED -- it needs " << total
        << " B of static shared memory against a " << kStaticShmemCap
        << " B cap (" << existingBytes
        << " B already live in the enclosing kernel + " << raiseBytes
        << " B for the raise: " << breakdown
        << "). ptxas rejects the module outright at that size, so this "
           "is a capability refusal, not a cost decision; refusing to "
           "price a candidate the materializer could not emit.\n";
  return false;
}

} // namespace poseidon
