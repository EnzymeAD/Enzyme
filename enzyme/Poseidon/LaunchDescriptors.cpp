//=- LaunchDescriptors.cpp - launch-stub descriptor plumbing --------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LaunchDescriptors.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace llvm;

namespace poseidon {

bool writeDescriptor(StringRef cacheDir, StringRef wrapperName,
                     StringRef scheme, StringRef tag,
                     function_ref<void(raw_ostream &)> emitPayload) {
  (void)sys::fs::create_directories(cacheDir, true);
  SmallString<256> path(cacheDir);
  sys::path::append(path, wrapperName + scheme);
  std::error_code EC;
  raw_fd_ostream os(path, EC, sys::fs::OF_Text);
  if (EC) {
    errs() << tag << " cannot write " << path << ": " << EC.message() << "\n";
    return false;
  }
  os << wrapperName;
  emitPayload(os);
  os << '\n';
  return true;
}

void removeDescriptor(StringRef cacheDir, StringRef wrapperName,
                      StringRef scheme) {
  SmallString<256> path(cacheDir);
  sys::path::append(path, wrapperName + scheme);
  (void)sys::fs::remove(path.str());
}

void readDescriptors(StringRef cacheDir, StringRef scheme,
                     function_ref<void(ArrayRef<StringRef>)> parse) {
  std::error_code EC;
  for (sys::fs::directory_iterator it(cacheDir, EC), end; it != end && !EC;
       it.increment(EC)) {
    StringRef p = it->path();
    if (!p.ends_with(scheme))
      continue;
    auto buf = MemoryBuffer::getFile(p);
    if (!buf)
      continue;
    StringRef line = (*buf)->getBuffer().trim();
    SmallVector<StringRef, 8> toks;
    line.split(toks, ' ');
    parse(toks);
  }
}

StringRef mangledSuffix(StringRef mangled) {
  if (mangled.starts_with("_Z")) {
    mangled = mangled.drop_front(2);
    while (!mangled.empty() && mangled[0] >= '0' && mangled[0] <= '9')
      mangled = mangled.drop_front(1);
  }
  return mangled;
}

bool isLaunchStubFor(const Function &F, StringRef kernelName) {
  StringRef n = F.getName();
  return n == kernelName || (n.contains("__device_stub__") &&
                             n.ends_with(mangledSuffix(kernelName)));
}

} // namespace poseidon
