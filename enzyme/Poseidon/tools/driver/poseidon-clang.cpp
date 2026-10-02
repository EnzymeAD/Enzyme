//===- poseidon-clang.cpp - the Poseidon compiler driver -----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Three user actions, modelled on clang's PGO:
//
//   poseidon-clang++ -poseidon-profile-generate app.cu -o app
//   ./app small_problem
//   poseidon-clang++ -poseidon-profile-use -poseidon-tau=1e-7 app.cu -o app_opt
//
// The driver execs the clang the plugins were built against with both pass
// plugins in the order the pipeline needs, Poseidon's include directory, and
// the runtime libraries the selected action requires. Every other
// -poseidon-<flag> reaches the pass as -mllvm -poseidon-<flag>.
//
//===----------------------------------------------------------------------===//

#include "poseidon/poseidon.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <string>

using namespace llvm;

namespace {

std::string envOr(const char *Name, StringRef Fallback) {
  if (const char *V = ::getenv(Name))
    if (*V)
      return std::string(V);
  return Fallback.str();
}

// A file under the installation the driver was started from, falling back to
// the build-time location when the driver runs out of its build tree.
std::string relativeToPrefix(StringRef Prefix, StringRef Rel,
                             StringRef Fallback) {
  SmallString<256> P(Prefix);
  sys::path::append(P, Rel);
  if (sys::fs::exists(P))
    return std::string(P);
  return Fallback.str();
}

void printHelp(StringRef Prog) {
  outs()
      << Prog
      << ": clang with the Poseidon pass plugins loaded.\n\n"
         "  -poseidon-profile-generate    instrument the annotated sites and\n"
         "                                link the profiler runtime\n"
         "  -poseidon-profile-use[=<dir>] rewrite from a profile\n"
         "                                "
         "(default " POSEIDON_DEFAULT_PROFILE_DIR ")\n"
         "  -poseidon-tau=<rel>           accuracy target the rewrite must "
         "meet\n"
         "  -poseidon-confidence=<c>      fraction of the sampled inputs a\n"
         "                                matrix product must meet the target\n"
         "                                on (default 0.95)\n"
         "  -poseidon-cost-model=<csv>    cost model to price candidates "
         "with;\n"
         "                                without it the model for the target\n"
         "                                device is looked up "
         "(poseidon-calibrate)\n"
         "  -poseidon-kernels=<regex>     treat matching kernels as "
         "optimization\n"
         "                                sites ('all' matches every kernel)\n"
         "  -poseidon-cache=<dir>         Herbie results, DP tables, "
         "descriptors\n"
         "  -poseidon-report-path=<dir>   write the per-site reports\n"
         "  -poseidon-help                this message\n\n"
         "Any other -poseidon-<flag>[=<value>] is passed to the pass as\n"
         "-mllvm -poseidon-<flag>[=<value>]. POSEIDON_PLUGIN and "
         "ENZYME_PLUGIN\n"
         "override the plugin paths baked in at build time; POSEIDON_ECHO=1\n"
         "prints the clang command line this driver runs.\n";
}

bool isLanguageCXX(StringRef Lang) {
  return Lang == "cuda" || Lang == "c++" || Lang == "cu" ||
         Lang == "c++-header" || Lang == "cuda-cpp-output";
}

// An option whose value is a SEPARATE argv entry. The value must reach clang
// paired with its option and must never be read as a flag of its own: without
// this, a value that happens to spell a Poseidon flag is rewritten into an
// `-mllvm -poseidon-<x>` of its own and the pair is split, and a value that
// happens to end in `.cu` silently switches the driver into CUDA mode.
bool takesSeparateValue(StringRef A) {
  static const char *const Opts[] = {"-Xcuda-ptxas",
                                     "-Xcuda-nvlink",
                                     "-Xclang",
                                     "-Xlinker",
                                     "-Xassembler",
                                     "-Xpreprocessor",
                                     "-Xarch_device",
                                     "-Xarch_host",
                                     "-Xoffload-linker",
                                     "-mllvm",
                                     "-include",
                                     "-isystem",
                                     "-idirafter",
                                     "-iquote",
                                     "-o",
                                     "-x",
                                     "-T",
                                     "-u",
                                     "-z"};
  for (const char *O : Opts)
    if (A == O)
      return true;
  return false;
}

} // namespace

int main(int argc, char **argv) {
  std::string Exe = sys::fs::getMainExecutable(argv[0], (void *)(intptr_t)main);
  // argv[0], not the resolved binary: poseidon-clang++ is a symlink to
  // poseidon-clang and the name it was invoked under chooses the clang driver.
  StringRef Prog = sys::path::filename(argv[0]);
  SmallString<256> Prefix(sys::path::parent_path(sys::path::parent_path(Exe)));

  bool WantCXX = Prog.contains("++");
  bool ProfileGenerate = false, ProfileUse = false;
  bool IsCUDA = false, Links = true;
  std::string CudaPath = POSEIDON_DRIVER_CUDA_ROOT;

  SmallVector<std::string, 64> User;
  for (int I = 1; I < argc; ++I) {
    StringRef A = argv[I];

    if (A == "-poseidon-help" || A == "--poseidon-help") {
      printHelp(Prog);
      return 0;
    }

    StringRef Flag = A;
    if (Flag.consume_front("-") &&
        (Flag.consume_front("-poseidon-") || Flag.consume_front("poseidon-"))) {
      StringRef Name = Flag.split('=').first;
      if (Name == "profile-generate")
        ProfileGenerate = true;
      else if (Name == "profile-use")
        ProfileUse = true;
      // A caller that already wrote -mllvm in front of the flag gets what it
      // asked for; anyone else gets the -mllvm added here.
      if (User.empty() || User.back() != "-mllvm")
        User.push_back("-mllvm");
      User.push_back(("-poseidon-" + Flag).str());
      continue;
    }

    if (A == "-x" && I + 1 < argc) {
      StringRef Lang = argv[I + 1];
      IsCUDA |= Lang == "cuda" || Lang == "cu";
      WantCXX |= isLanguageCXX(Lang);
    } else if (A.starts_with("--cuda-gpu-arch=") || A == "-fcuda-rdc" ||
               A == "-fgpu-rdc") {
      IsCUDA = true;
    } else if (A.starts_with("--cuda-path=")) {
      IsCUDA = true;
      CudaPath = A.substr(strlen("--cuda-path=")).str();
    } else if (A == "-c" || A == "-S" || A == "-E" || A == "-fsyntax-only" ||
               A == "--cuda-device-only" || A == "--cuda-host-only") {
      Links = false;
    } else if (A.ends_with(".cu")) {
      IsCUDA = true;
    }
    User.push_back(A.str());
    if (takesSeparateValue(A) && I + 1 < argc) {
      User.push_back(argv[++I]);
    }
  }

  if (ProfileGenerate && ProfileUse) {
    errs() << Prog
           << ": -poseidon-profile-generate and -poseidon-profile-use are the "
              "two halves of the workflow; pass one\n";
    return 1;
  }

  std::string Clang = WantCXX ? POSEIDON_DRIVER_CLANGXX : POSEIDON_DRIVER_CLANG;
  std::string Plugin =
      envOr("POSEIDON_PLUGIN",
            relativeToPrefix(Prefix, "lib/" POSEIDON_DRIVER_PLUGIN_NAME,
                             POSEIDON_DRIVER_PLUGIN));
  std::string Enzyme = envOr("ENZYME_PLUGIN", POSEIDON_DRIVER_ENZYME);
  std::string Include =
      relativeToPrefix(Prefix, "include/poseidon/poseidon.h",
                       POSEIDON_DRIVER_INCLUDE "/poseidon/poseidon.h");
  Include =
      std::string(sys::path::parent_path(sys::path::parent_path(Include)));
  std::string LibDir =
      relativeToPrefix(Prefix, "lib/libposeidon_profile.a",
                       POSEIDON_DRIVER_LIBDIR "/libposeidon_profile.a");
  LibDir = std::string(sys::path::parent_path(LibDir));

  if (!sys::fs::exists(Plugin)) {
    errs() << Prog << ": Poseidon plugin not found at " << Plugin
           << "; set POSEIDON_PLUGIN\n";
    return 1;
  }
  if (Enzyme.empty() && ProfileGenerate) {
    errs() << Prog
           << ": -poseidon-profile-generate needs Enzyme to differentiate the "
              "probes; set ENZYME_PLUGIN to the pinned ClangEnzyme-*.so\n";
    return 1;
  }

  SmallVector<std::string, 128> Cmd;
  Cmd.push_back(Clang);
  // Before LLVM 22 clang parses -mllvm before loading -fpass-plugin, so the
  // plugins' options are known only if they are also loaded with -load.
  auto AddPlugin = [&](const std::string &P) {
    Cmd.push_back("-fpass-plugin=" + P);
#if LLVM_VERSION_MAJOR < 22
    Cmd.append({"-Xclang", "-load", "-Xclang", P});
#endif
  };
  AddPlugin(Plugin);
  if (!Enzyme.empty())
    AddPlugin(Enzyme);
  Cmd.push_back("-I" + Include);
  // Error-free transformations survive only if the compiler contracts within a
  // statement and never across one.
  Cmd.push_back("-ffp-contract=on");
  Cmd.append(User.begin(), User.end());

  if (ProfileGenerate && IsCUDA) {
    // The instrumented device code calls the profiler's device functions, so
    // the profiler is compiled into the application rather than linked.
    Cmd.push_back("-fcuda-rdc");
    Cmd.push_back(relativeToPrefix(
        Prefix, "share/poseidon/runtime/fpprofiler/FPProfilerCUDA.cu",
        POSEIDON_DRIVER_PROFILER_CU));
  }

  if (Links) {
    // clang links no CUDA runtime of its own, and every action here ends in a
    // device launch.
    if (IsCUDA && !CudaPath.empty()) {
      Cmd.push_back("-L" + CudaPath + "/lib64");
      Cmd.push_back("-lcudart");
    }
    Cmd.push_back("-L" + LibDir);
    // poseidon_metric, and the profiler the instrumented code calls into.
    Cmd.push_back("-lposeidon_profile");
    if (ProfileUse && IsCUDA) {
      // A rewrite may dispatch a matrix product to any of the three GEMM
      // runtimes; they export disjoint symbols and only the chosen one runs.
      Cmd.push_back("-lposeidon_rt");
      Cmd.push_back("-lcublas");
      StringRef CuMp = POSEIDON_DRIVER_CUMPSGEMM;
      if (!CuMp.empty()) {
        Cmd.push_back(("-L" + CuMp + "/build").str());
        Cmd.push_back("-lcumpsgemm");
        Cmd.push_back("-Xlinker");
        Cmd.push_back("-rpath");
        Cmd.push_back("-Xlinker");
        Cmd.push_back((CuMp + "/build").str());
      }
    }
  }

  SmallVector<StringRef, 128> Argv(Cmd.begin(), Cmd.end());
  if (::getenv("POSEIDON_ECHO")) {
    for (StringRef A : Argv)
      errs() << A << ' ';
    errs() << '\n';
  }

  std::string Err;
  int RC = sys::ExecuteAndWait(Clang, Argv, std::nullopt, {}, 0, 0, &Err);
  if (!Err.empty())
    errs() << Prog << ": " << Err << '\n';
  return RC;
}
