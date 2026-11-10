//=- LaunchDescriptors.cpp - launch-stub descriptor plumbing --------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LaunchDescriptors.h"
#include "FPProfileName.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace llvm;

namespace poseidon {

// A mangled kernel name can be longer than a file name may be (an MFEM forall
// lambda's runs past 300 characters), so the file is named by the profile
// name rule; the reader takes the full name from the file's first token.
static SmallString<256> descriptorPath(StringRef cacheDir,
                                       StringRef wrapperName,
                                       StringRef scheme) {
  SmallString<256> path(cacheDir);
  sys::path::append(path, profileNameStem(wrapperName.str()) + scheme.str());
  return path;
}

bool writeDescriptor(StringRef cacheDir, StringRef wrapperName,
                     StringRef scheme, StringRef tag,
                     function_ref<void(raw_ostream &)> emitPayload) {
  (void)sys::fs::create_directories(cacheDir, true);
  SmallString<256> path = descriptorPath(cacheDir, wrapperName, scheme);
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
  (void)sys::fs::remove(descriptorPath(cacheDir, wrapperName, scheme).str());
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
    // An internal-linkage name (a static kernel template, as MFEM's are).
    mangled.consume_front("L");
    while (!mangled.empty() && mangled[0] >= '0' && mangled[0] <= '9')
      mangled = mangled.drop_front(1);
  }
  return mangled;
}

StringRef withoutExternalizedPostfix(StringRef name) {
  for (StringRef postfix : {"__intern__", ".intern."}) {
    size_t at = name.rfind(postfix);
    if (at == StringRef::npos)
      continue;
    StringRef hash = name.drop_front(at + postfix.size());
    if (hash.empty() || !llvm::all_of(hash, llvm::isHexDigit))
      continue;
    return name.take_front(at);
  }
  return name;
}

bool isLaunchStubFor(const Function &F, StringRef kernelName) {
  StringRef n = F.getName();
  if (n == kernelName)
    return true;
  constexpr StringLiteral stubPrefix = "__device_stub__";
  if (!n.contains(stubPrefix))
    return false;
  StringRef kernel = withoutExternalizedPostfix(kernelName);
  if (n.ends_with(mangledSuffix(kernel)))
    return true;
  // A kernel in a namespace or a class: its identifier, and with it the
  // stub's prefix, sits inside the nested name rather than at its front.
  std::string stub = demangle(n);
  size_t at = stub.find(stubPrefix);
  if (at == std::string::npos || stub == n)
    return false;
  stub.erase(at, stubPrefix.size());
  return stub == demangle(kernel);
}

} // namespace poseidon
