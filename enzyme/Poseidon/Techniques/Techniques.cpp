//===- Techniques.cpp - the techniques of the pass ------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The one list of techniques, and for each the steps of a site's collection
// it takes part in (Core/Technique.h), written against its own entry points.
//
//===----------------------------------------------------------------------===//

#include "Core/Technique.h"
#include "Solve/Explain.h"
#include "Solve/Optimize.h"
#include "Solve/Session.h"
#include "Support/Options.h"
#include "Targets/Target.h"
#include "Techniques/Algebraic/Algebraic.h"
#include "Techniques/Matmul/Matmul.h"
#include "Techniques/Precision/Expansion.h"
#include "Techniques/Precision/Precision.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>

using namespace llvm;

namespace poseidon {

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

static void remarkAlgebraicUnavailable(Session &S, Function &F) {
  if (S.Opts.Candidates.AlgebraicLacksHerbie)
    remarkNoHerbie(S, F);
}

static Error proposeAlgebraic(Session &S, Function &F, SiteState &st,
                              Subgraph &subgraph, size_t index,
                              const SiteProfileData &) {
  return proposeHerbieCandidates(S, F, subgraph, int(index), st.herbieSymbols,
                                 st, st.algebraicUnits);
}

static void bindAlgebraic(Session &S, SiteState &st) {
  bindAll(S, st, st.algebraicUnits);
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
  st.expansionSafe = *safe;
  return Error::success();
}

static Error proposePrecision(Session &S, Function &F, SiteState &st,
                              Subgraph &subgraph, size_t,
                              const SiteProfileData &P) {
  return proposePrecisionCandidates(S, F, subgraph, st, P.header,
                                    st.expansionSafe, st.precisionUnits);
}

static void bindPrecision(Session &S, SiteState &st) {
  bindAll(S, st, st.precisionUnits);
}

// Matrix products on tensor cores or as library calls.

static bool matmulEnabled(const Options &O) {
  return O.Candidates.EnableMatmul;
}

// The site's matrix products, with the analyses their raise keeps in `st`.
static void recognizeMatmuls(Session &S, Function &F, const SiteProfileData &P,
                             SiteState &st) {
  const Options &O = S.Opts;
  const Target &T = Target::of(F);
  if (T.isDevice()) {
    st.raiseDT.emplace(F);
    st.raiseLI.emplace(*st.raiseDT);
    st.raiseAC.emplace(F);
    st.raiseTLII.emplace(Triple(F.getParent()->getTargetTriple()));
    st.raiseTLI.emplace(*st.raiseTLII, &F);
    st.raiseSE.emplace(F, *st.raiseTLI, *st.raiseAC, *st.raiseDT, *st.raiseLI);
    // The constant-trip products are the in-kernel raise's.
    if (T.hasInKernelMatrixRaise())
      findConstantShapeMatmuls(O, F, *st.raiseSE, *st.raiseLI, P.header,
                               st.abstractMatmuls);
    // Runtime-shape dense GEMMs. Runs after the constant-trip recognizer and
    // skips whatever that one claimed, so a product both recognize keeps the
    // constant-trip candidate set.
    findRuntimeShapeMatmuls(O, F, *st.raiseSE, *st.raiseLI, P.header, P.map,
                            st.abstractMatmuls);
  }

  if (O.Diagnostics.Print && !st.abstractMatmuls.empty()) {
    llvm::errs() << "[poseidon] Found " << st.abstractMatmuls.size()
                 << " matrix product(s) in " << F.getName() << '\n';
    auto dimStr = [](unsigned d) -> std::string {
      return d == ~0u ? std::string("?") : std::to_string(d);
    };
    for (const auto &m : st.abstractMatmuls) {
      llvm::errs() << "  Matmul[" << m.id << "]: " << dimStr(m.M) << 'x'
                   << dimStr(m.N) << 'x' << dimStr(m.K)
                   << " type=" << dtypeName(m.elemType) << " origin="
                   << (m.origin == RecognizedMatmul::Origin::RuntimeShape
                           ? "RuntimeShape"
                           : "ConstantShape")
                   << '\n';
    }
  }
}

static Error proposeMatmuls(Session &S, SiteState &st,
                            const SiteProfileData &P) {
  return generateMatmulCandidates(S, st.abstractMatmuls, P.map,
                                  st.matmulSampleLogBits, st.matmulUnits);
}

static void bindMatmuls(Session &S, SiteState &st) {
  bindAll(S, st, st.matmulUnits);
}

// clang-format off
static const Technique kTechniques[] = {
    {{UnitKind::Algebraic, "algebraic", "A", "an algebraic rewrite", true},
     algebraicEnabled, /*Recognize=*/nullptr, /*BeginSite=*/nullptr,
     remarkAlgebraicUnavailable, proposeAlgebraic, /*ProposeForSite=*/nullptr,
     bindAlgebraic, reportHerbieAnswers},
    {{UnitKind::Precision, "precision change", "P", "a precision change", true},
     precisionEnabled, /*Recognize=*/nullptr, beginPrecision,
     /*RemarkUnavailable=*/nullptr, proposePrecision,
     /*ProposeForSite=*/nullptr, bindPrecision, /*Report=*/nullptr},
    {{UnitKind::Matmul, "matrix product", "M", "a candidate", false},
     matmulEnabled, recognizeMatmuls, /*BeginSite=*/nullptr,
     /*RemarkUnavailable=*/nullptr, /*ProposeForSubgraph=*/nullptr,
     proposeMatmuls, bindMatmuls, /*Report=*/nullptr},
};
// clang-format on

ArrayRef<Technique> techniques() { return kTechniques; }

const UnitKindInfo &infoOf(UnitKind kind) {
  const UnitKindInfo &info = kTechniques[unsigned(kind)].Unit;
  assert(info.kind == kind && "techniques listed out of UnitKind order");
  return info;
}

} // namespace poseidon
