// Matmul candidate materialization: MatmulUnit::apply.
#include "HostDispatch.h"
#include "InKernelRaise.h"
#include "MatmulInternal.h"
#include "Pass/Session.h"
#include "RaiseWMMA.h"
#include "Sites/LaunchDescriptors.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace poseidon {

static const char *libraryCallFlagError(Scheme s) {
  switch (s) {
  case Scheme::Modular:
  case Scheme::Native:
    return "Ozaki-II candidate was selected but -poseidon-ozaki-host-dispatch "
           "is not set; it cannot materialize. (Candidate generation should "
           "not propose Ozaki-II without the host-dispatch flag.)";
  case Scheme::Split:
    return "TCEC dispatch was selected but -poseidon-ozaki-host-dispatch is "
           "not set; it cannot materialize. (Candidate generation should not "
           "propose it without the host-dispatch flag.)";
  case Scheme::Direct:
    return "direct reduced-precision dispatch was selected but "
           "-poseidon-ozaki-host-dispatch is not set; it cannot materialize. "
           "(Candidate generation should not propose it without the "
           "host-dispatch flag.)";
  }
  llvm_unreachable("unknown Scheme");
}

static const char *libraryCallDescriptorError(Scheme s) {
  switch (s) {
  case Scheme::Modular:
  case Scheme::Native:
    return "Ozaki-II was SELECTED for this GEMM but the host-dispatch "
           "descriptor could not be built (computeGemmBodyNote failed). The "
           "padded dispatch must materialize the picked rewrite; refusing to "
           "silently fall back to scalar FP64. Either the GEMM shape/layout is "
           "genuinely unsupported (then it must not be PROPOSED in candidate "
           "generation) or this is a descriptor-extraction bug.";
  case Scheme::Split:
    return "TCEC dispatch was SELECTED for this GEMM but the host-dispatch "
           "descriptor could not be built (computeGemmBodyNote failed). "
           "Refusing to silently fall back to scalar FP64.";
  case Scheme::Direct:
    return "direct reduced-precision dispatch was SELECTED for this GEMM but "
           "the host-dispatch descriptor could not be built "
           "(computeGemmBodyNote failed). Refusing to silently fall back to "
           "scalar FP64.";
  }
  llvm_unreachable("unknown Scheme");
}

static GemmBodyNote runtimeShapeNote(const AbstractMatmul &m) {
  const RuntimeGemmHandle &g = *m.hostGemm;
  GemmBodyNote note;
  note.cParam = g.cRef.param;
  note.aParam = g.aRef.param;
  note.bParam = g.bRef.param;
  note.cOffset = g.cRef.offset;
  note.aOffset = g.aRef.offset;
  note.bOffset = g.bRef.offset;
  note.assumedEqual = g.assumedEqual;
  note.standalone = true;
  note.valid = true;
  note.runtimeDims = true;
  note.beta = g.beta;
  // Profile-scale constants, written only as the descriptor's fallback: the
  // host rewrite evaluates the runtime forms below instead.
  note.N = m.globalK;
  note.gM = m.globalM;
  note.gNcols = m.globalN;
  note.gK = m.globalK;
  note.lda = (unsigned)(g.lda.isConst() ? g.lda.add : m.globalK);
  note.ldb = (unsigned)(g.ldb.isConst() ? g.ldb.add : m.globalK);
  note.ldc = (unsigned)(g.ldc.isConst() ? g.ldc.add : m.globalM);
  note.aColMajor = g.aColMajor;
  note.bColMajor = g.bColMajor;
  note.cColMajor = g.cColMajor;
  note.rM = g.M;
  note.rNcols = g.Ncols;
  note.rK = g.K;
  note.rlda = g.lda;
  note.rldb = g.ldb;
  note.rldc = g.ldc;
  return note;
}

// Records the launch descriptor that the host compile rewrites the launch
// from, fissioning a fused constant-trip body. The solver picked this
// candidate, so failing to record it is a hard error rather than a silent FP64
// fallback.
static void applyLibraryCall(Session &S, const AbstractMatmul &m,
                             const SchemeVariant &v) {
  Function *F = nullptr;
  GemmBodyNote note;
  switch (m.origin) {
  case AbstractMatmul::Origin::Invalid:
    report_fatal_error("Unexpected invalid AbstractMatmul origin");
  case AbstractMatmul::Origin::HostGemmLoopNest: {
    if (!S.Opts.Matmul.OzakiHostDispatch)
      report_fatal_error(
          "a runtime-shape host GEMM candidate was selected but "
          "-poseidon-ozaki-host-dispatch is not set; it has no other "
          "realization.");
    F = m.hostGemm->cStore->getFunction();
    // Host dispatch replaces the whole launch, so the body must contain nothing
    // but the GEMM. A fused body would need the device-side fission path, which
    // cannot rewrite an accumulating epilogue whose value does not flow through
    // the reduction loop; refuse instead of half-doing it.
    int argStores = 0;
    for (Instruction &I : instructions(*F))
      if (auto *S = dyn_cast<StoreInst>(&I))
        if (tracePointerRoot(S->getPointerOperand()).param >= 0)
          ++argStores;
    if (argStores != 1)
      report_fatal_error(
          "a runtime-shape host GEMM was selected but its body performs " +
          Twine(argStores) +
          " argument-rooted stores; only a pure GEMM body can be replaced by a "
          "library dispatch. Candidate generation should not have proposed "
          "it.");
    note = runtimeShapeNote(m);
    break;
  }
  case AbstractMatmul::Origin::ScalarLoopReduction:
    if (!S.Opts.Matmul.OzakiHostDispatch)
      report_fatal_error(libraryCallFlagError(v.scheme));
    F = m.scalarLoop.fma ? m.scalarLoop.fma->getFunction() : nullptr;
    if (!F || !computeGemmBodyNote(S.Opts, *F, m, note) || !note.valid)
      report_fatal_error(libraryCallDescriptorError(v.scheme));
    break;
  }
  note.scheme = descriptorScheme(v);
  note.numModuli = descriptorModulusField(v);
  noteGemmBody(S, F, note);
  if (!note.standalone)
    fissionGemmForDispatch(*F, m);
}

void MatmulUnit::apply(Session &S, size_t candidateIndex) {
  if (candidateIndex >= candidates.size())
    report_fatal_error("MatmulUnit index out of range");
  const Option &opt = candidates[candidateIndex];

  if (opt.realization == Realization::LibraryCall) {
    applyLibraryCall(S, *matmul, opt.variant);
    return;
  }
  switch (matmul->origin) {
  case AbstractMatmul::Origin::Invalid:
    report_fatal_error("Unexpected invalid AbstractMatmul origin");
  case AbstractMatmul::Origin::HostGemmLoopNest:
    report_fatal_error("a runtime-shape host GEMM was selected with a "
                       "scheme that has no host-dispatch realization");
  case AbstractMatmul::Origin::ScalarLoopReduction:
    break;
  }
  switch (opt.variant.scheme) {
  case Scheme::Direct:
    materializeScalarLoopRaise(*matmul, opt);
    return;
  case Scheme::Split:
    materializeOzakiIRaise(*matmul, opt);
    return;
  case Scheme::Modular:
  case Scheme::Native:
    break;
  }
  llvm_unreachable("no in-kernel realization of this scheme");
}

} // namespace poseidon
