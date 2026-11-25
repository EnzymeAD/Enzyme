//===- Site.h - the sites of a module -------------------------*- C++ -*-===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// By OptimizerEarly every way of writing a site has been lowered into one of
// three forms the IR carries: a marker call, an annotated kernel (or one
// -poseidon-kernels names), and the outlined body of a host function. The IR
// is what carries them from pass to pass and from an LTO pre-link compile to
// its post-link one; collectSites is the one place that reads them.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SITES_SITE_H
#define POSEIDON_SITES_SITE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace llvm {
class CallInst;
class Function;
class Module;
} // namespace llvm

namespace poseidon {

struct Options;

/// One place the compile optimizes: a marker call, an annotated kernel, or a
/// host function outlined into a body. The functions are the module's.
struct Site {
  /// How the site was written: a marker call, a kernel carrying the annotation
  /// or named by -poseidon-kernels, or an annotated host function.
  enum class Form { Marker, Kernel, HostBody };
  /// How this site was written.
  Form form;
  /// The marker's target (null when the marker names none), the kernel, or
  /// the host function's outlined body.
  llvm::Function *fn;
  /// The marker call of a Form::Marker site.
  llvm::CallInst *marker = nullptr;
  /// The site's own accuracy target, of a kernel's annotation or a host
  /// function's. A marker's is part of its argument list, read with it.
  std::optional<double> target;
  /// A kernel only -poseidon-kernels names.
  bool byRegex = false;

  /// Whether the site is a marker whose body is left to the late run.
  bool deferred() const;
};

/// The sites of M of the given forms: the forms in the order given, the sites
/// of one form in module order.
llvm::SmallVector<Site, 4> collectSites(const Options &O, llvm::Module &M,
                                        llvm::ArrayRef<Site::Form> forms);

/// The sites of M left to the late run: markers whose body deferSite deferred.
llvm::SmallVector<Site, 4> collectDeferredSites(llvm::Module &M);

} // namespace poseidon

#endif // POSEIDON_SITES_SITE_H
