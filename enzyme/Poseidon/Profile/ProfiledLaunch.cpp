//===- ProfiledLaunch.cpp - the launches of profiled kernels --------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProfiledLaunch.h"
#include "Enzyme/Utils.h"
#include "Instrument.h"
#include "Pass/Diagnostics.h"
#include "Pass/Options.h"
#include "Sites/Annotations.h"
#include "Sites/LaunchManifest.h"
#include "Support/Utils.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>

using namespace llvm;

namespace poseidon {

namespace {
struct ProfDesc {
  std::string kernel;
  int siteId = 0;
  unsigned nargs = 0;
  SmallVector<int, 8> ptrIdx;
  SmallVector<int, 8> seedWidth;
  /// Byte offset of each pointer inside its argument (0 for a pointer
  /// argument).
  SmallVector<int, 8> ptrOff;
};
} // namespace

static Error readProfDescriptors(const Options &O, Module &M,
                                 std::map<std::string, ProfDesc> &out) {
  return readDescriptors(
      O, M, LaunchKind::Profiling, [&](ArrayRef<StringRef> toks) -> Error {
        DescriptorCursor Cursor(toks);
        ProfDesc d;
        d.kernel = toks[0].str();
        d.siteId = Cursor.integer<int>("site id");
        d.nargs = Cursor.integer<unsigned>("argument count");
        unsigned np = Cursor.integer<unsigned>("pointer count");
        for (unsigned i = 0; i < np && Cursor.ok(); ++i)
          d.ptrIdx.push_back(Cursor.integer<int>("pointer argument"));
        for (unsigned i = 0; i < np && Cursor.ok(); ++i)
          d.seedWidth.push_back(Cursor.integer<int>("seed width"));
        d.ptrOff.assign(d.ptrIdx.size(), 0);
        if (Cursor.consume("off"))
          for (int &off : d.ptrOff) {
            off = Cursor.integer<int>("pointer offset");
            if (off < 0)
              Cursor.fail("its pointer offset " + Twine(off) + " is negative");
          }
        if (Error E = Cursor.takeError())
          return E;
        out[d.kernel] = d;
        return Error::success();
      });
}

static Constant *profIntArray(Module &M, const Twine &nameT,
                              ArrayRef<int> vals) {
  std::string name = nameT.str();
  Type *i32 = Type::getInt32Ty(M.getContext());
  SmallVector<Constant *, 8> elts;
  for (int v : vals)
    elts.push_back(ConstantInt::get(i32, (unsigned)v));
  ArrayType *AT = ArrayType::get(i32, elts.size());
  Constant *init = ConstantArray::get(AT, elts);
  if (auto *old = M.getNamedGlobal(name))
    if (old->getValueType() == AT && old->hasInitializer() &&
        old->getInitializer() == init)
      return old;
  auto *GV = new GlobalVariable(M, AT, /*isConstant=*/true,
                                GlobalValue::PrivateLinkage, init, name);
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  return GV;
}

Expected<bool> rewriteProfileStubBodies(const Options &O, Module &M) {
  std::map<std::string, ProfDesc> descs;
  if (Error E = readProfDescriptors(O, M, descs))
    return std::move(E);
  if (descs.empty())
    return false;

  LLVMContext &C = M.getContext();
  Type *ptr = getInt8PtrTy(C);
  Type *i32 = Type::getInt32Ty(C);
  Type *i64 = Type::getInt64Ty(C);
  FunctionCallee helper = M.getOrInsertFunction(
      "__poseidon_launch_profiled",
      FunctionType::get(
          Type::getVoidTy(C),
          {ptr, i32, ptr, ptr, i64, ptr, i32, ptr, i32, ptr, ptr, ptr}, false));

  bool changed = false;
  Error errors = Error::success();
  for (Function &F : M) {
    if (F.isDeclaration() || !F.getReturnType()->isVoidTy())
      continue;
    auto dIt = llvm::find_if(
        descs, [&](const auto &kv) { return isLaunchStubFor(F, kv.first); });
    if (dIt == descs.end())
      continue;
    const ProfDesc &d = dIt->second;
    // The pass reaches the host module through more than one extension point;
    // a stub that already calls the helper must not be rewritten again.
    if (callsFunctionNamed(F, "__poseidon_launch_profiled"))
      continue;
    // The launch's own argument array and configuration: what the stub's
    // parameters look like depends on the host ABI (an aggregate argument
    // arrives by value, by reference or split into registers), what the
    // launch is handed does not.
    CallInst *launch = findLaunchCall(F);
    CallInst *pop = findPopConfig(F);
    // Launched as written, the profiled kernel would read shadows nobody
    // passed it.
    if (!launch || !pop) {
      errors = joinErrors(
          std::move(errors),
          userError(F, F.getSubprogram(),
                    "the launch stub " + F.getName() +
                        " does not end in __cuda/__hipPopCallConfiguration "
                        "and cuda/hipLaunchKernel, so the profiling run "
                        "cannot pass its kernel the shadows it is profiled "
                        "with. Compile against the CUDA or ROCm headers, "
                        "which declare that launch API."));
      continue;
    }

    Constant *idxGV = profIntArray(M, "poseidon.ptr." + d.kernel, d.ptrIdx);
    Constant *offGV = profIntArray(M, "poseidon.off." + d.kernel, d.ptrOff);
    Constant *seedGV =
        profIntArray(M, "poseidon.seed." + d.kernel, d.seedWidth);

    IRBuilder<> B(launch);
    B.CreateCall(
        helper,
        {launchedKernel(*launch), ConstantInt::get(i32, (unsigned)d.siteId),
         poppedGrid(*pop), poppedBlock(*pop), launchSharedBytes(*launch),
         launchStream(*launch), ConstantInt::get(i32, d.nargs),
         launchArguments(*launch), ConstantInt::get(i32, d.ptrIdx.size()),
         idxGV, offGV, seedGV});
    launch->replaceAllUsesWith(Constant::getNullValue(launch->getType()));
    launch->eraseFromParent();
    if (O.Diagnostics.Print)
      errs() << "[poseidon] launch of " << d.kernel
             << " now allocates and seeds " << d.ptrIdx.size()
             << " shadow buffer(s)\n";
    changed = true;
  }
  if (errors)
    return std::move(errors);
  return changed;
}

bool registerProfiledKernels(const Options &O, Module &M) {
  bool changed = false;
  readProfileHeaders(O, M, [&](StringRef profileName, StringRef header) {
    changed |= registerProfileStatic(M, profileName, profileName, header);
  });
  return changed;
}

} // namespace poseidon
