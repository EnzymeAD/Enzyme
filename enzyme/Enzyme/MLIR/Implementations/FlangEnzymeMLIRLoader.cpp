//===- FlangEnzymeMLIRLoader.cpp - FlangEnzyme brings FlangEnzymeMLIR -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Built into FlangEnzyme-<v> (the LLVM pass plugin, `flang -fpass-plugin=`)
// when FlangEnzymeMLIR-<v> is built too: when flang loads FlangEnzyme, this
// loads FlangEnzymeMLIR next to it, as `flang -fc1 -load` would, so that the
// one -fpass-plugin brings in the FIR type annotations, which LLVM Enzyme's
// type analysis reads (see HLFIRFlangPluginRegistration.cpp). flang loads
// pass plugins before it parses -mllvm and -mmlir and before it builds its
// MLIR pipeline, so FlangEnzymeMLIR's options and pipeline callbacks are
// registered in time.
//
// Not its differentiation passes: with FlangEnzyme, Enzyme differentiates the
// LLVM IR, and the MLIR route would take the calls of f__enzyme_* (which it
// does not handle in every case yet). An explicit -load of FlangEnzymeMLIR,
// which flang does before it loads pass plugins, keeps them.
//
// FlangEnzymeMLIR is loaded rather than linked in, so that loading both
// plugins (-fpass-plugin=FlangEnzyme and -load FlangEnzymeMLIR) still loads
// one copy of it: its options would be registered twice otherwise, which
// LLVM rejects. Outside flang (e.g. `opt -load-pass-plugin`), FlangEnzymeMLIR
// does not load, as the FIR and flang symbols it needs are missing; nothing
// changes there.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <string>

namespace {

#if !defined(_WIN32)
// Where FlangEnzymeMLIR is, relative to this plugin: next to it once
// installed, under MLIR/Implementations in a build tree.
static std::string findFlangEnzymeMLIR() {
  Dl_info info;
  if (!dladdr(reinterpret_cast<void *>(&findFlangEnzymeMLIR), &info) ||
      !info.dli_fname)
    return "";
  llvm::SmallString<256> dir(info.dli_fname);
  llvm::sys::path::remove_filename(dir);
  for (const char *sub : {"", "MLIR/Implementations"}) {
    llvm::SmallString<256> path(dir);
    llvm::sys::path::append(path, sub, FLANG_ENZYME_MLIR_FILE);
    if (llvm::sys::fs::exists(path))
      return std::string(path);
  }
  return "";
}

struct LoadFlangEnzymeMLIR {
  LoadFlangEnzymeMLIR() {
    // Only in flang: FlangEnzymeMLIR registers itself with flang's pipeline
    // (fir::registerPassPipelineConfigCallback), which its static
    // initializer calls.
    if (!dlsym(RTLD_DEFAULT,
               "_ZN3fir34registerPassPipelineConfigCallbackESt8functionIFvR"
               "28MLIRToLLVMPassPipelineConfigEE"))
      return;
    std::string path = findFlangEnzymeMLIR();
    if (path.empty())
      return;
    // Loaded by a -load already: as the user asked for it, differentiation
    // passes and all.
    if (void *loaded = dlopen(path.c_str(), RTLD_LAZY | RTLD_NOLOAD)) {
      dlclose(loaded);
      return;
    }
    std::string error;
    // As flang loads a -load plugin.
    llvm::sys::DynamicLibrary lib =
        llvm::sys::DynamicLibrary::getPermanentLibrary(path.c_str(), &error);
    if (!lib.isValid())
      return;
    if (auto *typeAnnotationsOnly = reinterpret_cast<void (*)()>(
            lib.getAddressOfSymbol("enzymeFlangMLIRTypeAnnotationsOnly")))
      typeAnnotationsOnly();
  }
};
// Runs when flang loads FlangEnzyme as a pass plugin.
static LoadFlangEnzymeMLIR loadFlangEnzymeMLIR;
#endif

} // namespace
