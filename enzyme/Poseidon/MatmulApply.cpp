// Matmul candidate materialization: CandidateMatmul::apply.
#include "Flags.h"
#include "HostDispatch.h"
#include "InKernelRaise.h"
#include "RaiseWMMA.h"
#include "MatmulInternal.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace poseidon {

void CandidateMatmul::apply(size_t candidateIndex) {
  if (candidateIndex >= candidates.size())
    report_fatal_error("CandidateMatmul index out of range");
  const Option &opt = candidates[candidateIndex];

  switch (matmul->origin) {
  case AbstractMatmul::Origin::Invalid:
    report_fatal_error("Unexpected invalid AbstractMatmul origin");
  case AbstractMatmul::Origin::HostGemmLoopNest: {
    if (!flags::OzakiHostDispatch)
      report_fatal_error(
          "a runtime-shape host GEMM candidate was selected but "
          "-poseidon-ozaki-host-dispatch is not set; it has no other "
          "realization.");
    const RuntimeGemmHandle &g = *matmul->hostGemm;
    Function *F = g.cStore->getFunction();
    // Host dispatch replaces the whole launch, so the body must contain nothing
    // but the GEMM. A fused body would need the device-side fission path, which
    // cannot rewrite an accumulating epilogue whose value does not flow through
    // the reduction loop; refuse instead of half-doing it.
    int argStores = 0;
    for (Instruction &I : instructions(*F))
      if (auto *S = dyn_cast<StoreInst>(&I))
        if (traceToArgIndex(S->getPointerOperand()) >= 0)
          ++argStores;
    if (argStores != 1)
      report_fatal_error(
          "a runtime-shape host GEMM was selected but its body performs " +
          Twine(argStores) +
          " argument-rooted stores; only a pure GEMM body can be replaced by a "
          "library dispatch. Candidate generation should not have proposed "
          "it.");
    GemmBodyNote note;
    note.cParam = g.cParam;
    note.aParam = g.aParam;
    note.bParam = g.bParam;
    note.standalone = true;
    note.valid = true;
    note.runtimeDims = true;
    note.beta = g.beta;
    // Profile-scale constants, written only as the descriptor's fallback: the
    // host rewrite evaluates the runtime forms below instead.
    note.N = matmul->globalK;
    note.gM = matmul->globalM;
    note.gNcols = matmul->globalN;
    note.gK = matmul->globalK;
    note.lda = (unsigned)(g.lda.isConst() ? g.lda.add : matmul->globalK);
    note.ldb = (unsigned)(g.ldb.isConst() ? g.ldb.add : matmul->globalK);
    note.ldc = (unsigned)(g.ldc.isConst() ? g.ldc.add : matmul->globalM);
    note.aColMajor = g.aColMajor;
    note.bColMajor = g.bColMajor;
    note.cColMajor = g.cColMajor;
    auto rt = [](const RuntimeDim &d) {
      GemmBodyNote::RtDim r;
      r.param = d.param;
      r.mul = d.mul;
      r.add = d.add;
      return r;
    };
    note.rM = rt(g.M);
    note.rNcols = rt(g.Ncols);
    note.rK = rt(g.K);
    note.rlda = rt(g.lda);
    note.rldb = rt(g.ldb);
    note.rldc = rt(g.ldc);
    switch (opt.strategy) {
    case CandidateMatmul::Option::Strategy::OzakiII:
      note.scheme = DispatchScheme::OzakiII;
      note.numModuli = opt.strategyParam;
      break;
    case CandidateMatmul::Option::Strategy::TcecDispatch:
      note.scheme = DispatchScheme::Tcec;
      note.numModuli = opt.strategyParam;
      break;
    case CandidateMatmul::Option::Strategy::DirectDispatch:
      note.scheme = DispatchScheme::Direct;
      note.numModuli = opt.strategyParam; // operand format
      break;
    default:
      report_fatal_error("a runtime-shape host GEMM was selected with a "
                         "strategy that has no host-dispatch realization");
    }
    noteGemmBody(F, note);
    return;
  }
  case AbstractMatmul::Origin::ScalarLoopReduction:
    switch (opt.strategy) {
    case CandidateMatmul::Option::Strategy::Direct:
      materializeScalarLoopRaise(*matmul, opt);
      return;
    case CandidateMatmul::Option::Strategy::OzakiI:
      materializeOzakiIRaise(*matmul, opt);
      return;
    case CandidateMatmul::Option::Strategy::OzakiII:
      // Ozaki-II is realized as the host library dispatch: record the
      // descriptor at the solver-chosen nm and let the host pass rewrite the
      // launch (fused bodies are fissioned). The solver picked this candidate,
      // so failure is a hard error rather than a silent FP64 fallback.
      if (!flags::OzakiHostDispatch)
        report_fatal_error(
            "Ozaki-II candidate was selected but -poseidon-ozaki-host-dispatch "
            "is "
            "not set; it cannot materialize. (Candidate generation should not "
            "propose Ozaki-II without the host-dispatch flag.)");
      {
        llvm::Function *F = matmul->scalarLoop.fma
                                ? matmul->scalarLoop.fma->getFunction()
                                : nullptr;
        GemmBodyNote note;
        if (F && computeGemmBodyNote(*F, *matmul, note) && note.valid) {
          note.numModuli = opt.strategyParam; // solver-chosen num_moduli
          noteGemmBody(F, note);
          if (!note.standalone)
            fissionGemmForDispatch(*F, *matmul);
          return;
        }
      }
      report_fatal_error(
          "Ozaki-II was SELECTED for this GEMM but the host-dispatch "
          "descriptor "
          "could not be built (computeGemmBodyNote failed). The padded "
          "dispatch "
          "must materialize the picked rewrite; refusing to silently fall back "
          "to scalar FP64. Either the GEMM shape/layout is genuinely "
          "unsupported (then it must not be PROPOSED in candidate generation) "
          "or "
          "this is a descriptor-extraction bug.");
    case CandidateMatmul::Option::Strategy::TcecDispatch: {
      // Same path as Ozaki-II; only the scheme stamped on the descriptor
      // differs, selecting __poseidon_tcec_dgemm.
      if (!flags::OzakiHostDispatch)
        report_fatal_error(
            "TCEC dispatch was selected but -poseidon-ozaki-host-dispatch is "
            "not "
            "set; it cannot materialize. (Candidate generation should not "
            "propose it without the host-dispatch flag.)");
      llvm::Function *F = matmul->scalarLoop.fma
                              ? matmul->scalarLoop.fma->getFunction()
                              : nullptr;
      GemmBodyNote note;
      if (F && computeGemmBodyNote(*F, *matmul, note) && note.valid) {
        note.scheme = DispatchScheme::Tcec;
        note.numModuli = opt.strategyParam; // TCEC compute mode
        noteGemmBody(F, note);
        if (!note.standalone)
          fissionGemmForDispatch(*F, *matmul);
        return;
      }
      report_fatal_error(
          "TCEC dispatch was SELECTED for this GEMM but the host-dispatch "
          "descriptor could not be built (computeGemmBodyNote failed). "
          "Refusing to silently fall back to scalar FP64.");
    }
    case CandidateMatmul::Option::Strategy::DirectDispatch: {
      // Same path as the TCEC and Ozaki-II dispatches; the scheme stamped on
      // the descriptor selects __poseidon_direct_dgemm.
      if (!flags::OzakiHostDispatch)
        report_fatal_error(
            "direct reduced-precision dispatch was selected but "
            "-poseidon-ozaki-host-dispatch is not set; it cannot materialize. "
            "(Candidate generation should not propose it without the "
            "host-dispatch flag.)");
      llvm::Function *F = matmul->scalarLoop.fma
                              ? matmul->scalarLoop.fma->getFunction()
                              : nullptr;
      GemmBodyNote note;
      if (F && computeGemmBodyNote(*F, *matmul, note) && note.valid) {
        note.scheme = DispatchScheme::Direct;
        note.numModuli = opt.strategyParam; // operand format
        noteGemmBody(F, note);
        if (!note.standalone)
          fissionGemmForDispatch(*F, *matmul);
        return;
      }
      report_fatal_error(
          "direct reduced-precision dispatch was SELECTED for this GEMM but "
          "the host-dispatch descriptor could not be built "
          "(computeGemmBodyNote failed). Refusing to silently fall back to "
          "scalar FP64.");
    }
    }
    llvm_unreachable("unhandled strategy in ScalarLoopReduction apply");
  }
  llvm_unreachable("Unexpected AbstractMatmul origin");
}

} // namespace poseidon
