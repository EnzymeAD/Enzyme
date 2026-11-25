//===- Site.cpp - the sites of a module -----------------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Site.h"
#include "Annotations.h"
#include "Markers.h"
#include "Pass/Options.h"
#include "Support/Utils.h"
#include "Targets/Target.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Regex.h"

#include <cassert>

using namespace llvm;

namespace poseidon {

bool Site::deferred() const {
  return form == Form::Marker && fn && isDeferredBody(*fn);
}

static void collectMarkers(Module &M, SmallVectorImpl<Site> &out) {
  SmallVector<CallInst *, 4> calls;
  collectMarkerCalls(M, calls);
  for (CallInst *CI : calls)
    out.push_back({Site::Form::Marker, markerTargetOrNull(CI), CI});
}

/// The annotated kernels, then the kernels only -poseidon-kernels names, in
/// module order.
static void collectKernels(const Options &O, Module &M,
                           SmallVectorImpl<Site> &out) {
  const Target &T = Target::of(M);
  DenseMap<Function *, Site> found;
  for (auto &[fn, tau] : annotatedFunctions(M)) {
    if (fn->hasFnAttribute(kSiteLowered))
      continue;
    // A host function is a site of its own (outlineHostSites); a device
    // function makes the kernels that call it sites (propagateDeviceSites),
    // and what is left here no kernel of this translation unit calls.
    if (!T.isKernel(*fn)) {
      if (T.isDevice())
        warnSite(*fn, displayName(*fn) +
                          " is a device function no kernel of this "
                          "translation unit calls, so it is left as written; "
                          "on the device, put POSEIDON_OPTIMIZE on a kernel "
                          "or on a function a kernel calls");
      continue;
    }
    found.insert({fn, {Site::Form::Kernel, fn, nullptr, tau}});
  }

  if (!O.Profiling.Kernels.empty()) {
    // validateOptions refuses a pattern that is not a regular expression.
    bool all = O.Profiling.Kernels == "all";
    Regex re(O.Profiling.Kernels);
    for (Function &F : M)
      if (!F.isDeclaration() && !F.hasFnAttribute(kSiteLowered) &&
          T.isKernel(F) && (all || re.match(F.getName())))
        found.insert({&F,
                      {Site::Form::Kernel, &F, nullptr, std::nullopt,
                       /*byRegex=*/true}});
  }

  for (Function &F : M) {
    auto it = found.find(&F);
    if (it != found.end())
      out.push_back(it->second);
  }
}

static void collectHostBodies(Module &M, SmallVectorImpl<Site> &out) {
  for (Function &F : M) {
    if (F.empty() || !F.hasFnAttribute(kSiteBody))
      continue;
    std::optional<double> tau;
    StringRef value = F.getFnAttribute(kSiteBody).getValueAsString();
    if (!value.empty()) {
      double parsed = 0.0;
      [[maybe_unused]] const bool failed = value.getAsDouble(parsed);
      assert(!failed && "the pass writes a body's accuracy target as a number");
      tau = parsed;
    }
    out.push_back({Site::Form::HostBody, &F, nullptr, tau});
  }
}

SmallVector<Site, 4> collectDeferredSites(Module &M) {
  SmallVector<Site, 4> sites;
  collectMarkers(M, sites);
  erase_if(sites, [](const Site &site) { return !site.deferred(); });
  return sites;
}

SmallVector<Site, 4> collectSites(const Options &O, Module &M,
                                  ArrayRef<Site::Form> forms) {
  SmallVector<Site, 4> sites;
  for (Site::Form form : forms)
    switch (form) {
    case Site::Form::Marker:
      collectMarkers(M, sites);
      break;
    case Site::Form::Kernel:
      collectKernels(O, M, sites);
      break;
    case Site::Form::HostBody:
      collectHostBodies(M, sites);
      break;
    }
  return sites;
}

} // namespace poseidon
