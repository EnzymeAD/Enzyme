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
// -poseidon-<flag> reaches the pass as -Xclang -mllvm -Xclang -poseidon-<flag>.
//
//===----------------------------------------------------------------------===//

#include "poseidon/poseidon.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include <cstring>
#include <optional>
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
         "  -poseidon-tau=<rel>           accuracy target of every site that\n"
         "                                does not carry its own\n"
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
         "  -poseidon-explain=<dir>       write <func>.explain.json (see\n"
         "                                tools/explain/poseidon-explain)\n"
         "  -poseidon-help                this message\n\n"
         "Any other -poseidon-<flag>[=<value>], with or without -mllvm before\n"
         "it, reaches the pass as -Xclang -mllvm -Xclang -poseidon-<flag>.\n"
         "POSEIDON_PLUGIN and ENZYME_PLUGIN override the plugin paths baked\n"
         "in at build time; POSEIDON_ECHO=1 prints the clang command line\n"
         "this driver runs.\n";
}

bool isLanguageCXX(StringRef Lang) {
  return Lang == "cuda" || Lang == "c++" || Lang == "cu" ||
         Lang == "c++-header" || Lang == "cuda-cpp-output" || Lang == "hip";
}

// How many argv entries after `A` are its value: clang's driver options whose
// value can be the next entry (ClangSeparateOptions.def, as the clang the
// plugin is built against has them, the MultiArg ones, and the
// JoinedAndSeparate ones by their prefix). Such a value reaches clang paired
// with its option and is never read as a flag or an input of its own: a value
// that spells a Poseidon flag is not rewritten, and one that ends in .cu does
// not make the compile CUDA.
unsigned separateValues(StringRef A) {
  static const StringSet<> One = {
#define CLANG_SEPARATE_OPTION(Name) Name,
#define CLANG_SEPARATE_OPTION_IN(Name, First, Last)
#include "ClangSeparateOptions.def"
#undef CLANG_SEPARATE_OPTION
#undef CLANG_SEPARATE_OPTION_IN
  };
  static const StringMap<unsigned> Several = {
      {"-sectalign", 3}, {"-sectcreate", 3}, {"-sectobjectsymbols", 2},
      {"-sectorder", 3}, {"-segaddr", 2},    {"-segcreate", 3},
      {"-segprot", 3}};
  if (One.contains(A))
    return 1;
#define CLANG_SEPARATE_OPTION(Name)
#define CLANG_SEPARATE_OPTION_IN(Name, First, Last)                            \
  if (A == Name && First <= LLVM_VERSION_MAJOR &&                              \
      (Last == 0 || LLVM_VERSION_MAJOR <= Last))                               \
    return 1;
#include "ClangSeparateOptions.def"
#undef CLANG_SEPARATE_OPTION
#undef CLANG_SEPARATE_OPTION_IN
  for (StringRef Prefix : {"-Xarch_", "-Xoffload-compiler", "-Xoffload-linker",
                           "-Xopenmp-target="})
    if (A.starts_with(Prefix))
      return 1;
  return Several.lookup(A);
}

// Whether clang stops before the link when given `A` (the options of
// Driver::getFinalPhase), or the driver compiles one half of a GPU source.
bool stopsBeforeLink(StringRef A) {
  static const StringSet<> Stops = {"-E",
                                    "--preprocess",
                                    "-M",
                                    "--dependencies",
                                    "-MM",
                                    "--user-dependencies",
                                    "--precompile",
                                    "--precompile-reduced-bmi",
                                    "-extract-api",
                                    "-fmodule-header",
                                    "-fsyntax-only",
                                    "-print-supported-cpus",
                                    "--print-supported-cpus",
                                    "-mcpu=help",
                                    "-mtune=help",
                                    "-print-enabled-extensions",
                                    "--print-enabled-extensions",
                                    "-module-file-info",
                                    "-verify-pch",
                                    "-rewrite-objc",
                                    "-rewrite-legacy-objc",
                                    "--analyze",
                                    "-emit-cir",
                                    "-emit-ast",
                                    "-S",
                                    "--assemble",
                                    "-c",
                                    "--compile",
                                    "-emit-interface-stubs",
                                    "--cuda-device-only",
                                    "--cuda-host-only"};
  return Stops.contains(A) || A.starts_with("-fmodule-header=");
}

// The name and value of a Poseidon flag, -poseidon-<flag>[=<value>] or
// --poseidon-<flag>[=<value>].
std::optional<StringRef> poseidonFlag(StringRef A) {
  if (A.consume_front("-") &&
      (A.consume_front("-poseidon-") || A.consume_front("poseidon-")))
    return A;
  return std::nullopt;
}

// Whether the argument names the translation unit itself, written as a
// Poseidon flag or as the value of an option that forwards it.
bool namesUnit(StringRef A) {
  A.consume_front("-mllvm=");
  return A.consume_front("-") &&
         (A.consume_front("-poseidon-unit") ||
          A.consume_front("poseidon-unit")) &&
         (A.empty() || A.starts_with("="));
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
  bool IsCUDA = false, IsHIP = false, StopsBeforeLink = false, HasInput = false;
  std::string CudaPath = POSEIDON_DRIVER_CUDA_ROOT;
  std::string RocmPath = POSEIDON_DRIVER_ROCM_ROOT;

  SmallVector<std::string, 64> User;
  for (int I = 1; I < argc; ++I) {
    StringRef A = argv[I];

    if (A == "-poseidon-help" || A == "--poseidon-help") {
      printHelp(Prog);
      return 0;
    }

    // A Poseidon flag, bare or after clang's -mllvm. clang hands -Xclang to
    // each of its compile jobs and to nothing else, so the flag reaches every
    // compiler that loads the plugin and never a linker, on every target, and
    // a line that only links is not warned about it.
    std::optional<StringRef> Flag = poseidonFlag(A);
    if (!Flag && A == "-mllvm" && I + 1 < argc &&
        (Flag = poseidonFlag(argv[I + 1])))
      ++I;
    if (!Flag && A.starts_with("-mllvm="))
      Flag = poseidonFlag(A.substr(strlen("-mllvm=")));
    if (Flag) {
      StringRef Name = Flag->split('=').first;
      if (Name == "profile-generate")
        ProfileGenerate = true;
      else if (Name == "profile-use")
        ProfileUse = true;
      User.append(
          {"-Xclang", "-mllvm", "-Xclang", ("-poseidon-" + *Flag).str()});
      continue;
    }

    StringRef Lang;
    if (A == "-x" && I + 1 < argc)
      Lang = argv[I + 1];
    else if (A.starts_with("-x"))
      Lang = A.substr(strlen("-x"));
    StringRef Arch = A;
    if (!Lang.empty()) {
      IsCUDA |= Lang == "cuda" || Lang == "cu";
      IsHIP |= Lang == "hip";
      WantCXX |= isLanguageCXX(Lang);
    } else if (Arch.consume_front("--cuda-gpu-arch=") ||
               Arch.consume_front("--offload-arch=")) {
      // The processor tells the GPU; --cuda-gpu-arch is an alias of
      // --offload-arch, also for HIP.
      if (Arch.starts_with("gfx"))
        IsHIP = true;
      else if (Arch.starts_with("sm_") || A.starts_with("--cuda-gpu-arch="))
        IsCUDA = true;
    } else if (A == "-fcuda-rdc" || A == "-fgpu-rdc") {
      IsCUDA = true;
    } else if (A.starts_with("--cuda-path=")) {
      IsCUDA = true;
      CudaPath = A.substr(strlen("--cuda-path=")).str();
    } else if (A == "--hip-link") {
      IsHIP = true;
    } else if (A.starts_with("--rocm-path=")) {
      IsHIP = true;
      RocmPath = A.substr(strlen("--rocm-path=")).str();
    } else if (stopsBeforeLink(A)) {
      StopsBeforeLink = true;
    } else if (!A.starts_with("-") || A == "-") {
      HasInput = true;
      IsCUDA |= A.ends_with(".cu");
      IsHIP |= A.ends_with(".hip");
    }
    User.push_back(A.str());
    for (unsigned N = separateValues(A); N && I + 1 < argc; --N)
      User.push_back(argv[++I]);
  }

  // The host half of a compile takes the launch descriptors of the device half
  // that compiled the same source under the same unit name, so every compile
  // job of this invocation gets one, and no other invocation gets it.
  if (HasInput &&
      llvm::none_of(ArrayRef<char *>(argv + 1, argc - 1), namesUnit)) {
    SmallString<256> Cwd;
    (void)sys::fs::current_path(Cwd);
    std::string Invocation(Cwd);
    for (int I = 0; I < argc; ++I) {
      Invocation += '\0';
      Invocation += argv[I];
    }
    User.insert(User.begin(),
                {"-Xclang", "-mllvm", "-Xclang",
                 "-poseidon-unit=" + utohexstr(xxh3_64bits(Invocation),
                                               /*LowerCase=*/false,
                                               /*Width=*/16)});
  }

  // -fgpu-rdc is HIP's spelling too.
  if (IsHIP)
    IsCUDA = false;

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
  // Without an input clang compiles and links nothing (-v, --version,
  // -print-*), and gets the user's arguments alone.
  if (HasInput) {
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
    // Every HIP device compile is an LTO pre-link compile, in which Enzyme
    // leaves a derivative request to the post-link run unless told otherwise,
    // and the device link loads no plugin. The requests profile generation
    // emits, and a user's own request in a site, have to be lowered in the
    // compile itself. Ahead of the user's arguments, so that a later
    // -enzyme-lto-prelink wins.
    if (IsHIP && !Enzyme.empty())
      Cmd.append(
          {"-Xclang", "-mllvm", "-Xclang", POSEIDON_DRIVER_ENZYME_HIP_OPTION});
    Cmd.push_back("-I" + Include);
    // Error-free transformations survive only if the compiler contracts within
    // a statement and never across one.
    Cmd.push_back("-ffp-contract=on");
  }
  Cmd.append(User.begin(), User.end());

  if (HasInput && !StopsBeforeLink) {
    // clang links no CUDA runtime of its own, and every action here ends in a
    // device launch.
    if (IsCUDA && !CudaPath.empty()) {
      Cmd.push_back("-L" + CudaPath + "/lib64");
      Cmd.push_back("-lcudart");
    }
    if (IsHIP && !RocmPath.empty()) {
      Cmd.push_back("-L" + RocmPath + "/lib");
      Cmd.push_back("-lamdhip64");
    }
    Cmd.push_back("-L" + LibDir);
    // The shadows of a POSEIDON_OPTIMIZE function on the host are sized from
    // the allocations this library records.
    if (ProfileGenerate)
      Cmd.push_back("-lposeidon_profile_host");
    // The host half of the CUDA profiler (the pass links the device half into
    // each device module). Not keyed on CUDA inputs, so that a link of objects
    // alone gets it; a program that references none of it takes nothing. The
    // HIP build of it exports the same names, so a link takes one of the two:
    // HIP's for HIP inputs or where it is the only one built.
    bool HipProfiler = POSEIDON_DRIVER_PROFILER_HIP &&
                       (IsHIP || !POSEIDON_DRIVER_PROFILER_CUDA);
    if (ProfileGenerate && HipProfiler)
      Cmd.push_back("-lposeidon_profile_hip");
    else if (ProfileGenerate && POSEIDON_DRIVER_PROFILER_CUDA)
      Cmd.push_back("-lposeidon_profile_cuda");
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
    if (ProfileUse && IsHIP) {
      // Ozaki-II and the direct products on hipBLAS; TCEC has no HIP build.
      Cmd.push_back("-lposeidon_rt_hip");
      Cmd.push_back("-lhipblas");
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
