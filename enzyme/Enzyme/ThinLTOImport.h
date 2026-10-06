//===- ThinLTOImport.h - Bring functions to differentiate into ThinLTO ----===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Under ThinLTO, Enzyme differentiates in each post-link backend, which only
// sees its own module plus the functions the thin link imported into it. The
// function importer follows call edges of the module summary, but a function
// passed to __enzyme_autodiff is only referenced, so it is never imported.
//
// The pre-link half of this pass asks for those functions: it adds an anchor
// function whose function_entry_count metadata lists their GUIDs, which the
// module summary turns into critical call edges (the mechanism sample PGO uses
// to replay its profiled imports). The post-link half drops the anchor and
// makes an internal copy of each imported function to differentiate, and of
// the imported functions it calls, so the copies survive
// EliminateAvailableExternally until Enzyme runs.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_THINLTO_IMPORT_H
#define ENZYME_THINLTO_IMPORT_H

#include "PassUtils.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
}

/// At ThinLTO pre-link: request the import of every function passed to an
/// __enzyme_* call that this module only declares, and of the declarations
/// that functions it defines and differentiates call.
bool enzymeThinLTORequestImports(llvm::Module &M);

/// At ThinLTO post-link: drop the anchor the pre-link added, and give every
/// imported function to differentiate an internal copy.
bool enzymeThinLTOLocalizeImports(llvm::Module &M);

class EnzymeThinLTOImportPass final
    : public PassParent<EnzymeThinLTOImportPass> {
  friend PassParent<EnzymeThinLTOImportPass>;
  bool PostLink;
  static llvm::AnalysisKey Key;

public:
  explicit EnzymeThinLTOImportPass(bool PostLink) : PostLink(PostLink) {}
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
};

#endif
