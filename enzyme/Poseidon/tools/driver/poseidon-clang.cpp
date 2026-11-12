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
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
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

// `Rel` under the installation prefix `Prefix`.
std::string underPrefix(StringRef Prefix, const Twine &Rel) {
  SmallString<256> P(Prefix);
  sys::path::append(P, Rel);
  return std::string(P);
}

// The toolkit a GPU runtime is linked from: the one given, else the one
// Poseidon was built against (none if it was built without), which has to be
// on this machine.
std::optional<std::string> toolkitRoot(StringRef Prog, StringRef Toolkit,
                                       StringRef Given, StringRef Built,
                                       StringRef How) {
  if (!Given.empty() || Built.empty() || sys::fs::is_directory(Built))
    return Given.empty() ? Built.str() : Given.str();
  errs() << Prog << ": the " << Toolkit << " Poseidon was built against, "
         << Built << ", is not on this machine; " << How << "\n";
  return std::nullopt;
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

  // Response files are expanded first, with the quoting clang reads them with,
  // so that a Poseidon flag in one is converted and the file is no input.
  BumpPtrAllocator Alloc;
  SmallVector<const char *, 64> Args(argv, argv + argc);
  if (Error E = cl::ExpansionContext(Alloc, cl::TokenizeGNUCommandLine)
                    .expandResponseFiles(Args)) {
    errs() << Prog << ": " << toString(std::move(E)) << '\n';
    return 1;
  }

  bool WantCXX = Prog.contains("++");
  bool ProfileGenerate = false, ProfileUse = false;
  bool IsCUDA = false, IsHIP = false, StopsBeforeLink = false, HasInput = false;
  StringRef CudaPathArg, RocmPathArg, HipPathArg;

  SmallVector<std::string, 64> User;
  for (size_t I = 1; I < Args.size(); ++I) {
    StringRef A = Args[I];

    if (A == "-poseidon-help" || A == "--poseidon-help") {
      printHelp(Prog);
      return 0;
    }

    // A Poseidon flag, bare or after clang's -mllvm. clang hands -Xclang to
    // each of its compile jobs and to nothing else, so the flag reaches every
    // compiler that loads the plugin and never a linker, on every target, and
    // a line that only links is not warned about it.
    std::optional<StringRef> Flag = poseidonFlag(A);
    if (!Flag && A == "-mllvm" && I + 1 < Args.size() &&
        (Flag = poseidonFlag(Args[I + 1])))
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
    if (A == "-x" && I + 1 < Args.size())
      Lang = Args[I + 1];
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
      CudaPathArg = A.substr(strlen("--cuda-path="));
    } else if (A == "--hip-link") {
      IsHIP = true;
    } else if (A.starts_with("--rocm-path=")) {
      IsHIP = true;
      RocmPathArg = A.substr(strlen("--rocm-path="));
    } else if (A.starts_with("--hip-path=")) {
      IsHIP = true;
      HipPathArg = A.substr(strlen("--hip-path="));
    } else if (stopsBeforeLink(A)) {
      StopsBeforeLink = true;
    } else if (!A.starts_with("-") || A == "-") {
      HasInput = true;
      IsCUDA |= A.ends_with(".cu");
      IsHIP |= A.ends_with(".hip");
    }
    User.push_back(A.str());
    for (unsigned N = separateValues(A); N && I + 1 < Args.size(); --N)
      User.push_back(Args[++I]);
  }

  // The host half of a compile takes the launch descriptors of the device half
  // that compiled the same source under the same unit name, so every compile
  // job of this invocation gets one, and no other invocation gets it.
  if (HasInput &&
      llvm::none_of(ArrayRef<const char *>(Args).drop_front(), namesUnit)) {
    SmallString<256> Cwd;
    (void)sys::fs::current_path(Cwd);
    std::string Invocation(Cwd);
    for (const char *A : Args) {
      Invocation += '\0';
      Invocation += A;
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
  // The HIP runtime's libraries, from where clang takes the HIP runtime:
  // --hip-path, HIP_PATH, --rocm-path, ROCM_PATH, else the ROCm Poseidon was
  // built against.
  std::string RocmPath = HipPathArg.str();
  if (RocmPath.empty())
    RocmPath = envOr("HIP_PATH", RocmPathArg);
  if (RocmPath.empty())
    RocmPath = envOr("ROCM_PATH", "");

  if (ProfileGenerate && ProfileUse) {
    errs() << Prog
           << ": -poseidon-profile-generate and -poseidon-profile-use are the "
              "two halves of the workflow; pass one\n";
    return 1;
  }

  // The clang the plugin is built against, wherever the driver runs, and the
  // plugins, header and runtimes of the build tree the driver runs from or of
  // its installation, never some of each. Enzyme installs its plugins to lib/.
  std::string Clang = WantCXX ? POSEIDON_DRIVER_CLANGXX : POSEIDON_DRIVER_CLANG;
  StringRef BuiltEnzyme = POSEIDON_DRIVER_ENZYME;
  std::string Plugin = POSEIDON_DRIVER_PLUGIN, Enzyme = BuiltEnzyme.str();
  std::string Include = POSEIDON_DRIVER_INCLUDE;
  std::string LibDir = POSEIDON_DRIVER_LIBDIR;
  if (!sys::fs::equivalent(sys::path::parent_path(Exe),
                           POSEIDON_DRIVER_BINDIR)) {
    Plugin = underPrefix(Prefix, POSEIDON_DRIVER_INSTALL_LIBDIR
                         "/" POSEIDON_DRIVER_PLUGIN_NAME);
    if (!BuiltEnzyme.empty())
      Enzyme = underPrefix(Prefix, "lib/" + sys::path::filename(BuiltEnzyme));
    Include = underPrefix(Prefix, POSEIDON_DRIVER_INSTALL_INCLUDEDIR);
    LibDir = underPrefix(Prefix, POSEIDON_DRIVER_INSTALL_LIBDIR);
  }
  Plugin = envOr("POSEIDON_PLUGIN", Plugin);
  if (!sys::fs::exists(Plugin)) {
    errs() << Prog << ": Poseidon plugin not found at " << Plugin
           << "; set POSEIDON_PLUGIN\n";
    return 1;
  }
  if (std::string Given = envOr("ENZYME_PLUGIN", ""); !Given.empty()) {
    Enzyme = Given;
  } else if (!Enzyme.empty() && !sys::fs::exists(Enzyme)) {
    errs() << Prog << ": Enzyme plugin not found at " << Enzyme
           << "; install Enzyme to the same prefix or set ENZYME_PLUGIN\n";
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
    // A system directory, as the CMake package gives it: the header's
    // diagnostics are not the user's.
    Cmd.append({"-isystem", Include});
    // Error-free transformations survive only if the compiler contracts within
    // a statement and never across one.
    Cmd.push_back("-ffp-contract=on");
  }
  Cmd.append(User.begin(), User.end());

  if (HasInput && !StopsBeforeLink) {
    // clang links no CUDA runtime of its own, and every action here ends in a
    // device launch.
    std::optional<std::string> CudaRoot =
        IsCUDA
            ? toolkitRoot(Prog, "CUDA toolkit", CudaPathArg,
                          POSEIDON_DRIVER_CUDA_ROOT, "pass --cuda-path=<dir>")
            : "";
    std::optional<std::string> RocmRoot =
        IsHIP ? toolkitRoot(Prog, "ROCm", RocmPath, POSEIDON_DRIVER_ROCM_ROOT,
                            "pass --rocm-path=<dir> or set ROCM_PATH")
              : "";
    if (!CudaRoot || !RocmRoot)
      return 1;
    if (!CudaRoot->empty()) {
      Cmd.push_back("-L" + *CudaRoot + "/lib64");
      Cmd.push_back("-lcudart");
    }
    if (!RocmRoot->empty()) {
      Cmd.push_back("-L" + *RocmRoot + "/lib");
      Cmd.push_back("-lamdhip64");
    }
    Cmd.push_back("-L" + LibDir);
    // The shadows of a POSEIDON_OPTIMIZE function on the host are sized from
    // the allocations this library records.
    if (ProfileGenerate)
      Cmd.push_back("-lposeidon_profile_host");
    // The host half of the GPU profiler (the pass links the device half into
    // each device module): HIP's for HIP, CUDA's for CUDA, and for a link of
    // objects alone, which names no GPU, the one this build has; a program
    // that references none of it takes nothing. The two export the same names,
    // so a link takes one of them, and never the other GPU's.
    bool HipProfiler = IsHIP || (!IsCUDA && !POSEIDON_DRIVER_PROFILER_CUDA);
    if (ProfileGenerate && HipProfiler && POSEIDON_DRIVER_PROFILER_HIP)
      Cmd.push_back("-lposeidon_profile_hip");
    else if (ProfileGenerate && !HipProfiler && POSEIDON_DRIVER_PROFILER_CUDA)
      Cmd.push_back("-lposeidon_profile_cuda");
    // poseidon_metric, and the profiler the instrumented code calls into.
    Cmd.push_back("-lposeidon_profile");
    if (ProfileUse && IsCUDA) {
      // A rewrite may dispatch a matrix product to any of the three GEMM
      // runtimes; they export disjoint symbols and only the chosen one runs.
      Cmd.push_back("-lposeidon_rt");
      Cmd.push_back("-lcublas");
      // What the runtime links besides the toolkit's libraries, each found
      // again where it was built when the program runs, after cuBLAS:
      // cuMpSGEMM defines cuBLAS's GEMM entry points. Handed to the linker
      // alone: after the user's -x cuda clang would compile a file input.
      SmallVector<StringRef, 2> RtLibs;
      StringRef(POSEIDON_DRIVER_RT_LIBS).split(RtLibs, ',', -1, false);
      for (StringRef Lib : RtLibs) {
        Cmd.push_back("-Xlinker");
        Cmd.push_back(Lib.str());
        Cmd.push_back("-Xlinker");
        Cmd.push_back("-rpath");
        Cmd.push_back("-Xlinker");
        Cmd.push_back(sys::path::parent_path(Lib).str());
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

  // An expanded response file can make the command longer than the system
  // takes: clang then reads it from one of the driver's own, written as clang
  // writes its response files.
  SmallString<128> ResponseFile;
  std::string ResponseArg;
  if (!sys::commandLineFitsWithinSystemLimits(Clang, Argv)) {
    int FD;
    if (std::error_code EC = sys::fs::createTemporaryFile(
            "poseidon-clang", "rsp", FD, ResponseFile)) {
      errs() << Prog << ": cannot create a response file: " << EC.message()
             << '\n';
      return 1;
    }
    raw_fd_ostream OS(FD, /*shouldClose=*/true);
    for (StringRef A : ArrayRef<StringRef>(Argv).drop_front()) {
      OS << '"';
      for (char C : A) {
        if (C == '"' || C == '\\')
          OS << '\\';
        OS << C;
      }
      OS << "\" ";
    }
    OS.close();
    if (OS.has_error()) {
      errs() << Prog << ": cannot write " << ResponseFile << ": "
             << OS.error().message() << '\n';
      OS.clear_error();
      sys::fs::remove(ResponseFile);
      return 1;
    }
    ResponseArg = ("@" + ResponseFile).str();
    StringRef Program = Argv.front();
    Argv.assign({Program, StringRef(ResponseArg)});
  }

  std::string Err;
  int RC = sys::ExecuteAndWait(Clang, Argv, std::nullopt, {}, 0, 0, &Err);
  if (!Err.empty())
    errs() << Prog << ": " << Err << '\n';
  if (!ResponseFile.empty())
    sys::fs::remove(ResponseFile);
  return RC;
}
