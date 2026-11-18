//===- FlangEnzymeMLIRLoader.cpp - FlangEnzyme brings FlangEnzymeMLIR -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Loads FlangEnzymeMLIR when flang loads the FlangEnzyme pass plugin, so that
// `flang -fpass-plugin=FlangEnzyme-<v>.so` alone also adds Enzyme's passes to
// flang's MLIR pipeline. The library is loaded rather than linked in so that
// an additional `-load` of it shares one copy and registers its options once.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <functional>
#include <string>

#if !defined(_WIN32)
/// Returns the path of FlangEnzymeMLIR next to this plugin (install tree) or
/// under MLIR/Flang (build tree), or an empty string if it is in neither.
static std::string findFlangEnzymeMLIR() {
  Dl_info info;
  if (!dladdr(reinterpret_cast<void *>(&findFlangEnzymeMLIR), &info) ||
      !info.dli_fname)
    return "";
  llvm::SmallString<256> dir(info.dli_fname);
  llvm::sys::path::remove_filename(dir);
  for (const char *sub : {"", "MLIR/Flang"}) {
    llvm::SmallString<256> path(dir);
    llvm::sys::path::append(path, sub, FLANG_ENZYME_MLIR_FILE);
    if (llvm::sys::fs::exists(path))
      return std::string(path);
  }
  return "";
}

struct MLIRToLLVMPassPipelineConfig;
namespace fir {
/// Defined by flang; null when FlangEnzyme is loaded by another host.
__attribute__((weak)) void registerPassPipelineConfigCallback(
    std::function<void(MLIRToLLVMPassPipelineConfig &)>);
} // namespace fir

namespace {
struct LoadFlangEnzymeMLIR {
  LoadFlangEnzymeMLIR() {
    // FlangEnzymeMLIR cannot load into a host without flang's pipeline.
    if (!&fir::registerPassPipelineConfigCallback)
      return;
    std::string path = findFlangEnzymeMLIR();
    if (path.empty())
      return;
    // An earlier `-load` already registered its passes and options.
    if (void *loaded = dlopen(path.c_str(), RTLD_LAZY | RTLD_NOLOAD)) {
      dlclose(loaded);
      return;
    }
    std::string error;
    llvm::sys::DynamicLibrary::getPermanentLibrary(path.c_str(), &error);
  }
};
} // namespace

static LoadFlangEnzymeMLIR loadFlangEnzymeMLIR;
#endif
