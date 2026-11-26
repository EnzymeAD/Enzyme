//===- Techniques.cpp - the techniques of the pass ------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The one list of techniques, and for each the steps it takes part in
// (Core/Technique.h), written against its own entry points and state.
//
//===----------------------------------------------------------------------===//

#include "Core/Technique.h"
#include "Solve/Explain.h"
#include "Solve/Optimize.h"
#include "Solve/Session.h"
#include "Support/CacheKey.h"
#include "Support/Diagnostics.h"
#include "Support/Options.h"
#include "Targets/Target.h"
#include "Techniques/Algebraic/Algebraic.h"
#include "Techniques/Algebraic/HerbieRun.h"
#include "Techniques/Matmul/Matmul.h"
#include "Techniques/Precision/Expansion.h"
#include "Techniques/Precision/Narrowing.h"
#include "Techniques/Precision/Precision.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;

namespace poseidon {

TechniqueState::~TechniqueState() = default;

// Binds the site's units held in `units`, in order, and lists them.
template <typename UnitT>
static void bindAll(Session &S, SiteState &st, SmallVectorImpl<UnitT> &units) {
  for (size_t i = 0; i < units.size(); ++i) {
    units[i].bind(S, st, st, i);
    st.units.push_back(&units[i]);
  }
}

// Algebraic rewrites from Herbie.

static bool algebraicEnabled(const Options &O) {
  return O.Candidates.EnableAlgebraic;
}

// The missed-optimization remark, once per compile, that no algebraic rewrite
// is proposed because no Herbie can run.
static void remarkAlgebraicUnavailable(Session &S, Function &F) {
  bool &reported = S.get<HerbieRunState>().LackReported;
  if (!S.Opts.Candidates.AlgebraicLacksHerbie || reported)
    return;
  reported = true;
  const std::string program = herbieProgram(S.Opts.Herbie);
  OptimizationRemarkEmitter ORE(&F);
  ORE.emit([&]() {
    OptimizationRemarkMissed R(kRemarkPass, "NoHerbie", siteLocation(S, F),
                               &F.getEntryBlock());
    R << "Poseidon proposes no algebraic rewrites: "
      << (program.empty() ? kBuiltWithoutHerbie.str()
                          : "there is no Herbie at " + program);
    return R;
  });
}

static Error proposeAlgebraic(Session &S, Function &F, SiteState &st,
                              Subgraph &subgraph, size_t index,
                              const SiteProfileData &) {
  AlgebraicSiteState &A = st.get<AlgebraicSiteState>();
  return proposeHerbieCandidates(S, F, subgraph, int(index), A.symbols, st,
                                 A.units);
}

static void bindAlgebraic(Session &S, SiteState &st) {
  bindAll(S, st, st.get<AlgebraicSiteState>().units);
}

// Precision changes and floating-point expansions.

static bool precisionEnabled(const Options &O) {
  return O.Candidates.EnablePrecision;
}

static Error beginPrecision(Session &S, Function &F, SiteState &st) {
  Expected<bool> safe =
      expansionsRoundedAsWritten(S.Opts, F, !st.subgraphs.empty());
  if (!safe)
    return safe.takeError();
  st.get<PrecisionSiteState>().expansionSafe = *safe;
  return Error::success();
}

static Error proposePrecision(Session &S, Function &F, SiteState &st,
                              Subgraph &subgraph, size_t,
                              const SiteProfileData &P) {
  PrecisionSiteState &PS = st.get<PrecisionSiteState>();
  return proposePrecisionCandidates(S, F, subgraph, st, P.header,
                                    PS.expansionSafe, PS.units);
}

static void bindPrecision(Session &S, SiteState &st) {
  bindAll(S, st, st.get<PrecisionSiteState>().units);
}

// The subgraphs as written, each output counted once: a precision unit's
// baseline covers every output of its subgraph, so an algebraic unit's counts
// only on a subgraph without one. `unpriced` counts the algebraic units, and
// the outputs of the precision units, left out for want of a price.
static double subgraphBaseline(SiteState &st, unsigned &unpriced) {
  SmallPtrSet<const Subgraph *, 4> subgraphsWithPrecision;
  double baseline = 0.0;
  for (const PrecisionUnit &PU : st.get<PrecisionSiteState>().units) {
    baseline += PU.initialAccCost;
    subgraphsWithPrecision.insert(PU.subgraph);
    if (!PU.unpricedReason.empty())
      unpriced += PU.subgraph->outputs.size();
  }
  SmallPtrSet<const Value *, 8> arrayPriced;
  for (const AlgebraicUnit &RU : st.get<AlgebraicSiteState>().units) {
    if (subgraphsWithPrecision.count(RU.subgraph))
      continue;
    // An algebraic unit with no candidates was never priced, so its share of
    // the baseline is unknown; it cannot appear in a solution either.
    if (RU.candidates.empty()) {
      ++unpriced;
      continue;
    }
    // An array core prices all of its outputs at once; each output's baseline
    // is counted once however many cores cover it.
    if (llvm::any_of(RU.roots,
                     [&](const Value *V) { return arrayPriced.contains(V); }))
      continue;
    if (RU.isArray())
      arrayPriced.insert(RU.roots.begin(), RU.roots.end());
    baseline += RU.initialAccCost;
  }
  return baseline;
}

// Narrows the shared staging and the FP64 PHIs of F that the applied precision
// changes left reading narrower values, as pricing did.
static void narrowApplied(Session &S, Function &F) {
  narrowSharedStaging(S, F, /*announce=*/true, ParamNarrowing::Both, F);
  demoteFPCastPHIs(F);
}

// Matrix products on tensor cores or as library calls.

static bool matmulEnabled(const Options &O) {
  return O.Candidates.EnableMatmul;
}

// The site's matrix products, with the analyses their raise keeps.
static bool recognizeMatmuls(Session &S, Function &F, const SiteProfileData &P,
                             SiteState &st) {
  const Options &O = S.Opts;
  const Target &T = Target::of(F);
  MatmulSiteState &M = st.get<MatmulSiteState>();
  if (T.isDevice()) {
    M.DT.emplace(F);
    M.LI.emplace(*M.DT);
    M.AC.emplace(F);
    M.TLII.emplace(Triple(F.getParent()->getTargetTriple()));
    M.TLI.emplace(*M.TLII, &F);
    M.SE.emplace(F, *M.TLI, *M.AC, *M.DT, *M.LI);
    // The constant-trip products are the in-kernel raise's.
    if (T.hasInKernelMatrixRaise())
      findConstantShapeMatmuls(O, F, *M.SE, *M.LI, P.header, M.matmuls);
    // Runtime-shape dense GEMMs. Runs after the constant-trip recognizer and
    // skips whatever that one claimed, so a product both recognize keeps the
    // constant-trip candidate set.
    findRuntimeShapeMatmuls(O, F, *M.SE, *M.LI, P.header, P.map, M.matmuls);
  }

  if (O.Diagnostics.Print && !M.matmuls.empty()) {
    llvm::errs() << "[poseidon] Found " << M.matmuls.size()
                 << " matrix product(s) in " << F.getName() << '\n';
    auto dimStr = [](unsigned d) -> std::string {
      return d == ~0u ? std::string("?") : std::to_string(d);
    };
    for (const auto &m : M.matmuls) {
      llvm::errs() << "  Matmul[" << m.id << "]: " << dimStr(m.M) << 'x'
                   << dimStr(m.N) << 'x' << dimStr(m.K)
                   << " type=" << dtypeName(m.elemType) << " origin="
                   << (m.origin == RecognizedMatmul::Origin::RuntimeShape
                           ? "RuntimeShape"
                           : "ConstantShape")
                   << '\n';
    }
  }
  return !M.matmuls.empty();
}

static Error proposeMatmuls(Session &S, SiteState &st,
                            const SiteProfileData &P) {
  const Options &O = S.Opts;
  MatmulSiteState &M = st.get<MatmulSiteState>();
  // The operand sampling of the matrix accuracy model, 0 for none. Uniform
  // sampling over [min, max] under-prices fixed-point Ozaki on matrices of
  // wide dynamic range, so a site solved under a target defaults to a wide
  // log-uniform range.
  const double target = st.tau > 0.0 ? st.tau : (double)O.Solver.Tau;
  const unsigned sampleLogBits =
      O.Matmul.SampleLogBits.value_or(target > 0.0 ? 40 : 0);
  return generateMatmulCandidates(S, M.matmuls, P.map, sampleLogBits, M.units);
}

static void bindMatmuls(Session &S, SiteState &st) {
  bindAll(S, st, st.get<MatmulSiteState>().units);
}

// Two builds of one source differ in whether the error-corrected library call
// is a candidate.
static void keyTcecBackend(CacheKey &key) {
  key.add("TCEC backend", kHaveTcecRuntime ? "cuMpSGEMM" : "none");
}

// clang-format off
static const Technique kTechniques[] = {
    {AlgebraicUnit::Info, algebraicEnabled, /*Recognize=*/nullptr,
     /*BeginSite=*/nullptr, remarkAlgebraicUnavailable, proposeAlgebraic,
     /*ProposeForSite=*/nullptr, bindAlgebraic, /*AddToTableKey=*/nullptr,
     /*Baseline=*/nullptr, /*SiteApplied=*/nullptr, /*SiteLaunched=*/nullptr,
     reportHerbieAnswers, /*RewriteHost=*/nullptr},
    {PrecisionUnit::Info, precisionEnabled, /*Recognize=*/nullptr,
     beginPrecision, /*RemarkUnavailable=*/nullptr, proposePrecision,
     /*ProposeForSite=*/nullptr, bindPrecision, /*AddToTableKey=*/nullptr,
     subgraphBaseline, narrowApplied, /*SiteLaunched=*/nullptr,
     /*Report=*/nullptr, /*RewriteHost=*/nullptr},
    {MatmulUnit::Info, matmulEnabled, recognizeMatmuls, /*BeginSite=*/nullptr,
     /*RemarkUnavailable=*/nullptr, /*ProposeForSubgraph=*/nullptr,
     proposeMatmuls, bindMatmuls, keyTcecBackend, /*Baseline=*/nullptr,
     writePendingLibraryCall, noteLibraryCallLaunch, /*Report=*/nullptr,
     rewriteGemmStubBodies},
};
// clang-format on

ArrayRef<Technique> techniques() { return kTechniques; }

} // namespace poseidon
