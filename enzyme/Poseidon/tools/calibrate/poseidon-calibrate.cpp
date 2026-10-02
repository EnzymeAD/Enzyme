//===- poseidon-calibrate.cpp - measure a device's cost model ------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Poseidon prices every candidate from one CSV measured on the device the code
// will run on. This tool measures it, and is to Poseidon what llvm-profdata is
// to clang PGO: run it once per GPU, and every later compile finds the model by
// the module's target-cpu.
//
// Five arms write disjoint row families into the same CSV:
//   microbm    per-op saturated reciprocal throughputs and the WMMA tile rows,
//              plus the "# native_arch=" header the compiler matches on
//   ozaki      ozaki_dispatch_rel,nm*   host-dispatched Ozaki-II GEMM
//   tcec       tcec_dispatch_rel        host-dispatched error-corrected GEMM
//   direct     direct_dispatch_rel      host-dispatched reduced-precision GEMM
//   inkernel   wmma_inkernel_rel        tensor-core raises inside a kernel
// Each dispatch row is the candidate's wall clock over the scalar-FP64 GEMM's
// on this device, so all four families are comparable in one unit.
//
// A sixth arm, herbie-platform, measures nothing: it translates the CSV into
// the Herbie platform file (<csv>.herbie.rkt) that the algebraic search ranks
// its rewrites by, so Herbie and the DP price the same device. It needs no
// GPU, so `--only herbie-platform --out <csv>` runs anywhere.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>

using namespace llvm;

namespace {

cl::opt<std::string> GpuId("gpu", cl::init(""), cl::value_desc("id"),
                           cl::desc("Device to measure (default: the first of "
                                    "CUDA_VISIBLE_DEVICES, else 0)"));
cl::opt<std::string>
    Out("out", cl::init(""), cl::value_desc("path"),
        cl::desc("Cost model to write; a directory takes cm_<arch>_<gpu>.csv "
                 "(default: $XDG_CACHE_HOME/poseidon)"));
cl::opt<bool> Check("check",
                    cl::desc("Validate the resolved cost model against this "
                             "device instead of measuring"));
cl::opt<bool> Force("force",
                    cl::desc("Re-measure families that are already present"));
cl::opt<std::string>
    Only("only", cl::init(""), cl::value_desc("arms"),
         cl::desc("Comma-separated subset of "
                  "microbm,ozaki,tcec,direct,inkernel,herbie-platform"));
cl::opt<std::string> ProfileDir(
    "profile", cl::init(""), cl::value_desc("dir"),
    cl::desc("Profile the in-kernel arm enumerates its candidates from "
             "(default: the ozp surrogate in the artifact)"));
cl::opt<std::string>
    Herbie("herbie", cl::init(POSEIDON_CALIBRATE_HERBIE),
           cl::value_desc("path"),
           cl::desc("Herbie binary the generated platform must load into "
                    "(default: the one this build was configured with)"));
cl::opt<std::string> CuMpSGEMM(
    "cumpsgemm", cl::init(POSEIDON_CALIBRATE_CUMPSGEMM), cl::value_desc("dir"),
    cl::desc("cuMpSGEMM checkout whose build/ the TCEC arm is measured "
             "against"));
cl::opt<unsigned> RefN("n", cl::init(2048), cl::value_desc("N"),
                       cl::desc("Square reference shape the dispatch arms and "
                                "the in-kernel arm are timed at"));
cl::opt<unsigned> Reps("reps", cl::init(30), cl::value_desc("count"),
                       cl::desc("Timed repetitions per dispatch measurement"));

std::string ToolDir;

std::string sibling(StringRef Name) {
  SmallString<256> P(ToolDir);
  sys::path::append(P, Name);
  return std::string(P);
}

// share/poseidon of the installation this tool was started from, and the
// source tree it was built from when it runs out of its build directory. The
// two layouts carry the same names below that point.
std::string shipped(StringRef Rel) {
  SmallString<256> P(sys::path::parent_path(ToolDir));
  sys::path::append(P, "share/poseidon", Rel);
  if (sys::fs::exists(P))
    return std::string(P);
  SmallString<256> S(POSEIDON_CALIBRATE_SRCDIR);
  sys::path::append(S, Rel);
  return std::string(S);
}

std::string num(double V, const char *Fmt = "%.6f") {
  char B[64];
  snprintf(B, sizeof(B), Fmt, V);
  return std::string(B);
}

[[noreturn]] void fail(const Twine &Msg) {
  errs() << "poseidon-calibrate: " << Msg << '\n';
  std::exit(1);
}

// Runs a program, optionally capturing its stdout into a file.
int run(ArrayRef<StringRef> Argv, StringRef StdoutFile = "",
        StringRef StderrFile = "") {
  std::optional<StringRef> Redirects[3] = {std::nullopt, std::nullopt,
                                           std::nullopt};
  if (!StdoutFile.empty())
    Redirects[1] = StdoutFile;
  if (!StderrFile.empty())
    Redirects[2] = StderrFile;
  std::string Err;
  int RC =
      sys::ExecuteAndWait(Argv[0], Argv, std::nullopt, Redirects, 0, 0, &Err);
  if (!Err.empty())
    fail(Argv[0] + ": " + Err);
  return RC;
}

std::string readFile(StringRef Path) {
  auto Buf = MemoryBuffer::getFile(Path);
  if (!Buf)
    fail("cannot read " + Path + ": " + Buf.getError().message());
  return (*Buf)->getBuffer().str();
}

void writeFile(StringRef Path, StringRef Content) {
  std::error_code EC;
  raw_fd_ostream OS(Path, EC);
  if (EC)
    fail("cannot write " + Path + ": " + EC.message());
  OS << Content;
}

std::string capture(ArrayRef<StringRef> Argv) {
  SmallString<128> Tmp;
  if (sys::fs::createTemporaryFile("poseidon-calibrate", "out", Tmp))
    fail("cannot create a temporary file");
  int RC = run(Argv, Tmp);
  std::string S = RC == 0 ? readFile(Tmp) : std::string();
  sys::fs::remove(Tmp);
  return S;
}

bool hasFamily(StringRef Csv, StringRef Family) {
  for (StringRef Line : llvm::split(Csv, '\n'))
    if (Line.starts_with(Family) &&
        Line.drop_front(Family.size()).starts_with(","))
      return true;
  return false;
}

// Replaces every row of the families the new rows belong to, so a re-measured
// family never leaves a stale row behind.
void mergeRows(std::string &Csv, StringRef Rows) {
  SmallVector<StringRef, 4> Families;
  for (StringRef Line : llvm::split(Rows, '\n')) {
    Line = Line.trim();
    if (Line.empty())
      continue;
    StringRef Fam = Line.split(',').first;
    if (!llvm::is_contained(Families, Fam))
      Families.push_back(Fam);
  }
  std::string Kept;
  for (StringRef Line : llvm::split(Csv, '\n')) {
    if (Line.empty())
      continue;
    StringRef Fam = Line.split(',').first;
    if (llvm::is_contained(Families, Fam))
      continue;
    Kept += Line;
    Kept += '\n';
  }
  for (StringRef Line : llvm::split(Rows, '\n')) {
    Line = Line.trim();
    if (Line.empty())
      continue;
    Kept += Line;
    Kept += '\n';
  }
  Csv = Kept;
}

struct Device {
  std::string Id;
  std::string Name; // punctuation stripped, as it appears in the file name
  std::string Arch; // sm_<cc>
};

Device queryDevice() {
  Device D;
  D.Id = GpuId;
  if (D.Id.empty()) {
    if (const char *V = ::getenv("CUDA_VISIBLE_DEVICES"))
      D.Id = StringRef(V).split(',').first.str();
    if (D.Id.empty())
      D.Id = "0";
  }
  auto Smi = sys::findProgramByName("nvidia-smi");
  if (!Smi)
    fail("nvidia-smi not found, so this device cannot be identified");
  std::string O = capture({*Smi, "-i", D.Id, "--query-gpu=name,compute_cap",
                           "--format=csv,noheader"});
  StringRef Line = StringRef(O).split('\n').first.trim();
  if (Line.empty())
    fail("no CUDA device with id " + D.Id);
  StringRef NameRef, CC;
  std::tie(NameRef, CC) = Line.split(',');
  for (char C : NameRef.trim())
    if (isAlnum(C))
      D.Name += C;
  // The names carry a vendor prefix that says nothing about the device.
  StringRef N(D.Name);
  for (StringRef Prefix : {"NVIDIA", "GeForce", "Tesla", "Quadro"})
    while (N.consume_front(Prefix))
      ;
  D.Name = N.str();
  StringRef Major, Minor;
  std::tie(Major, Minor) = CC.trim().split('.');
  D.Arch = ("sm_" + Major + Minor).str();
  return D;
}

std::string defaultOutDir() {
  if (const char *X = ::getenv("XDG_CACHE_HOME"))
    if (*X)
      return (StringRef(X) + "/poseidon").str();
  SmallString<128> Home;
  if (!sys::path::home_directory(Home))
    fail("neither XDG_CACHE_HOME nor a home directory is set; pass --out");
  sys::path::append(Home, ".cache/poseidon");
  return std::string(Home);
}

std::string resolveCsv(const Device &D) {
  std::string Path = Out;
  if (Path.empty())
    Path = defaultOutDir();
  if (sys::fs::is_directory(Path) || StringRef(Path).ends_with("/") ||
      !StringRef(sys::path::filename(Path)).ends_with(".csv")) {
    if (std::error_code EC = sys::fs::create_directories(Path))
      fail("cannot create " + Path + ": " + EC.message());
    SmallString<256> P(Path);
    sys::path::append(P, "cm_" + D.Arch + "_" + D.Name + ".csv");
    return std::string(P);
  }
  sys::fs::create_directories(sys::path::parent_path(Path));
  return Path;
}

bool wants(StringRef Arm) {
  if (Only.empty())
    return true;
  SmallVector<StringRef, 8> Arms;
  StringRef(Only).split(Arms, ',', -1, false);
  return llvm::is_contained(Arms, Arm);
}

// ---- the arms -------------------------------------------------------------

// The Herbie platform generated from a CSV lives next to it, named after it:
// one platform per cost model, so a directory holding two devices' models
// cannot mix them up.
std::string platformPathFor(StringRef Csv) {
  StringRef Stem = Csv.ends_with(".csv") ? Csv.drop_back(4) : Csv;
  return Stem.str() + ".herbie.rkt";
}

std::string csvNativeArch(const std::string &Csv) {
  StringRef Arch;
  for (StringRef Line : llvm::split(readFile(Csv), '\n'))
    if (Line.consume_front("# native_arch="))
      Arch = Line.trim();
  return Arch.str();
}

// Herbie's algebraic search ranks candidates by a platform cost table. This
// arm writes that table from the same CSV the DP prices from, so a rewrite is
// proposed and priced on one device's numbers. No GPU is touched: the CSV is
// the only input.
void armHerbiePlatform(const std::string &Csv) {
  std::string Arch = csvNativeArch(Csv);
  if (Arch.empty())
    fail(Csv + " carries no '# native_arch=' header, so no Herbie platform "
               "can be generated from it");
  if (!StringRef(Arch).starts_with("sm_")) {
    outs() << "[calibrate] herbie-platform: " << Arch
           << " is a host cost model; Herbie keeps its own default platform\n";
    return;
  }
  std::string Script = shipped("tools/herbie/csv_to_herbie_platform.py");
  if (!sys::fs::exists(Script))
    fail("the platform generator is missing: " + Script);
  std::string Python = POSEIDON_CALIBRATE_PYTHON;
  if (Python.empty() || !sys::fs::exists(Python)) {
    auto P = sys::findProgramByName("python3");
    if (!P)
      fail("python3 is needed to generate the Herbie platform from " + Csv +
           "; put one on PATH or configure the build with a Python3 "
           "interpreter");
    Python = *P;
  }
  // The CSV file name carries the device (cm_<arch>_<device>.csv); it is a
  // comment in the platform header, so a generated file says what it is.
  std::string Device;
  StringRef Name = sys::path::stem(Csv);
  if (Name.consume_front("cm_" + Arch + "_"))
    Device = Name.str();
  std::string Out = platformPathFor(Csv);
  SmallVector<StringRef, 14> Argv = {Python,   Script, "--csv",    Csv,
                                     "--arch", Arch,   "--output", Out};
  if (!Herbie.empty()) {
    Argv.push_back("--herbie-binary");
    Argv.push_back(Herbie);
  }
  if (!Device.empty()) {
    Argv.push_back("--device");
    Argv.push_back(Device);
  }
  if (run(Argv))
    fail("the Herbie platform generator failed on " + Csv);
  outs() << "[calibrate] wrote " << Out << '\n';
}

void armMicrobm(const std::string &Csv) {
  if (sys::fs::exists(Csv) && !Force) {
    outs() << "[calibrate] per-op throughputs already measured: " << Csv
           << '\n';
    return;
  }
  outs() << "[calibrate] measuring per-op throughput (several minutes)\n";
  std::string Rows = capture({sibling("poseidon-microbm")});
  if (!StringRef(Rows).contains("\nfmul,double,") ||
      !StringRef(Rows).contains("# native_arch="))
    fail("the microbenchmark produced no usable cost model");
  writeFile(Csv, Rows);
  outs() << "[calibrate] wrote " << Csv << '\n';
}

void writeRows(const std::string &Csv, StringRef Arm, StringRef Family,
               const std::string &Rows) {
  if (!hasFamily(Rows, Family))
    fail(Arm + " calibration produced no " + Family + " row");
  std::string Content = readFile(Csv);
  mergeRows(Content, Rows);
  writeFile(Csv, Content);
  for (StringRef Line : llvm::split(Rows, '\n'))
    if (!Line.trim().empty())
      outs() << "  " << Line.trim() << '\n';
}

bool alreadyMeasured(const std::string &Csv, StringRef Arm, StringRef Family) {
  if (Force || !hasFamily(readFile(Csv), Family))
    return false;
  outs() << "[calibrate] " << Arm << ": already measured\n";
  return true;
}

void armDispatch(const std::string &Csv, StringRef Arm, StringRef Family,
                 StringRef Tool) {
  if (alreadyMeasured(Csv, Arm, Family))
    return;
  outs() << "[calibrate] measuring " << Arm << " host dispatch\n";
  writeRows(Csv, Arm, Family,
            capture({sibling(Tool), utostr(RefN), utostr(Reps)}));
}

// The error-corrected dispatch has a backend: the built-in cuBLAS kernels, or
// the cuMpSGEMM library the paper measured. Which one a device is faster with
// is a measurement, so the caller names the library and the calibrator is
// rebuilt against it rather than assumed.
void armTcec(const std::string &Csv, const Device &D) {
  if (alreadyMeasured(Csv, "tcec", "tcec_dispatch_rel"))
    return;
  if (CuMpSGEMM.empty()) {
    outs() << "[calibrate] measuring tcec host dispatch (built-in backend)\n";
    writeRows(Csv, "tcec", "tcec_dispatch_rel",
              capture({sibling("poseidon-tcec-calibrate"), utostr(RefN),
                       utostr(Reps)}));
    return;
  }
  auto Nvcc = sys::findProgramByName("nvcc", {POSEIDON_CALIBRATE_CUDA_BIN});
  if (!Nvcc)
    fail("nvcc is needed to measure the TCEC arm against cuMpSGEMM");
  SmallString<256> WorkBuf;
  sys::fs::createUniqueDirectory("poseidon-tcec", WorkBuf);
  std::string Bin = std::string(WorkBuf) + "/tcec_calibrate";
  std::string Lib = CuMpSGEMM + "/build";
  outs() << "[calibrate] measuring tcec host dispatch against " << Lib << '\n';
  if (run({*Nvcc, "-O3", "-arch=" + D.Arch, "-std=c++17",
           shipped("tools/calibrate/tcec_calibrate.cu"),
           shipped("runtime/tcec/tcec_rt.cu"), "-DPOSEIDON_TCEC_USE_CUMPSGEMM",
           "-I" + CuMpSGEMM + "/include", "-L" + Lib, "-lcumpsgemm", "-lcublas",
           "-o", Bin},
          std::string(WorkBuf) + "/build.log",
          std::string(WorkBuf) + "/build.log"))
    fail("the TCEC calibrator did not build against " + CuMpSGEMM + "; see " +
         std::string(WorkBuf) + "/build.log");
  std::string LdPath = Lib;
  if (const char *P = ::getenv("LD_LIBRARY_PATH"))
    LdPath += std::string(":") + P;
  ::setenv("LD_LIBRARY_PATH", LdPath.c_str(), 1);
  writeRows(Csv, "tcec", "tcec_dispatch_rel",
            capture({Bin, utostr(RefN), utostr(Reps)}));
}

// The in-kernel classes are the one family the compiler has to be asked about:
// each is a rewrite the pass proposes, so the harness compiles its kernel once
// per class through the plugin and times what a real solve would emit. A class
// the pass cannot propose or build here is reported and left absent.
void armInKernel(const std::string &Csv, const Device &D) {
  std::string Content = readFile(Csv);
  std::string Src = shipped("tools/calibrate/inkernel_calibrate.cu");
  if (!sys::fs::exists(Src))
    fail("the in-kernel harness source is missing: " + Src);

  std::string Prof = ProfileDir;
  if (Prof.empty())
    Prof = POSEIDON_CALIBRATE_INKERNEL_PROFILE;
  SmallString<256> KernelProf(Prof);
  sys::path::append(
      KernelProf, "preprocess__Z10matmul_optPdPKdS1__poseidon_body.fpprofile");
  if (!sys::fs::exists(KernelProf))
    fail("the in-kernel arm needs the surrogate profile carrying " +
         Twine(sys::path::filename(KernelProf)) +
         "; produce it (artifacts/cgo2027/benchmarks/ozp: "
         "scripts/profile_only.sh) and pass --profile <dir>");

  SmallString<256> WorkBuf;
  sys::fs::createUniqueDirectory("poseidon-inkernel", WorkBuf);
  std::string Work(WorkBuf);
  outs() << "[calibrate] in-kernel workdir " << Work << '\n';

  std::string Clang = sibling("poseidon-clang++");
  std::string Arch = "--cuda-gpu-arch=" + D.Arch;
  std::string CalN = "-DCAL_N=" + utostr(RefN);
  std::string ProfUse = "-poseidon-profile-use=" + Prof;
  std::string CostModel = "-poseidon-cost-model=" + Csv;
  std::string Cache = "-poseidon-cache=" + Work + "/cache";
  std::string BaseExe = Work + "/base.exe";
  std::string BaseLog = Work + "/build_base.log";
  std::string EnumLog = Work + "/enumerate.log";

  // The scalar-FP64 baseline: the denominator every rel row shares. Compiled
  // by the same driver, with no action asked of the pass, so the only
  // difference from a candidate is the rewrite.
  if (run({Clang, "-x", "cuda", Arch, "-O3", "-Wno-unknown-cuda-version", CalN,
           Src, "-o", BaseExe},
          BaseLog, BaseLog))
    fail("the scalar-FP64 baseline did not build; see " + BaseLog);
  Regex TimeRe("time_ms = ([0-9.]+)");
  auto timeOf = [&](StringRef Exe) -> double {
    std::string O = capture({Exe, utostr(Reps)});
    SmallVector<StringRef, 2> M;
    if (!TimeRe.match(O, &M))
      return 0.0;
    double V = 0;
    M[1].getAsDouble(V);
    return V;
  };
  double BaseMs = timeOf(BaseExe);
  if (BaseMs <= 0)
    fail("the scalar-FP64 baseline produced no time");
  outs() << "[calibrate] scalar-FP64 baseline " << num(BaseMs, "%.4f")
         << " ms\n";

  // One compile prints the candidate table; -poseidon-apply-rewrites is
  // mandatory under -poseidon-inkernel-calibration, so a placeholder price can
  // never reach a DP solve.
  SmallVector<std::string, 32> Common = {Clang,
                                         "-x",
                                         "cuda",
                                         Arch,
                                         "-O3",
                                         "-Wno-unknown-cuda-version",
                                         CalN,
                                         ProfUse,
                                         Cache,
                                         CostModel,
                                         "-poseidon-enable-herbie=0",
                                         "-poseidon-enable-pt=0",
                                         "-poseidon-loose-coverage",
                                         "-poseidon-raise-wmma",
                                         "-poseidon-print",
                                         "-poseidon-inkernel-calibration"};
  auto compile = [&](const std::string &Apply, const std::string &Exe,
                     const std::string &Log) {
    SmallVector<StringRef, 32> Argv(Common.begin(), Common.end());
    Argv.push_back(Apply);
    Argv.push_back(Src);
    Argv.push_back("-o");
    Argv.push_back(Exe);
    return run(Argv, Log, Log);
  };
  compile("-poseidon-apply-rewrites=M0_0", "/dev/null", EnumLog);

  // "[poseidon]   #8 tcec n=2 wmma m16n16k8 tf32/f32  compCost/MAC=..."
  Regex Cand("^\\[poseidon\\][ ]+#([0-9]+) (.*)  compCost/MAC");
  Regex Direct("^wmma m[0-9]+n[0-9]+k[0-9]+ ([a-z0-9]+)/([a-z0-9]+)$");
  Regex Tcec(
      "^tcec n=([0-9]+) wmma m[0-9]+n[0-9]+k[0-9]+ ([a-z0-9]+)/([a-z0-9]+)$");
  std::map<std::string, std::string> Classes; // class -> candidate index
  for (StringRef Line : llvm::split(readFile(EnumLog), '\n')) {
    SmallVector<StringRef, 3> M;
    if (!Cand.match(Line, &M))
      continue;
    StringRef Idx = M[1], Label = M[2].trim();
    SmallVector<StringRef, 4> P;
    std::string Cls;
    if (Direct.match(Label, &P))
      Cls = (P[1] + "_" + P[2]).str();
    else if (Tcec.match(Label, &P))
      Cls = ("tcec_n" + P[1] + "_" + P[2] + "_" + P[3]).str();
    else
      continue; // a host-dispatch family: priced by its own arm
    Classes.emplace(Cls, Idx.str());
  }
  if (Classes.empty())
    fail("no in-kernel class was proposed on this device; see " + EnumLog);
  outs() << "[calibrate] " << Classes.size()
         << " in-kernel class(es) proposed\n";

  std::string Rows;
  for (const auto &[Cls, Idx] : Classes) {
    if (!Force && hasFamily(Content, "wmma_inkernel_rel") &&
        StringRef(Content).contains("wmma_inkernel_rel," + Cls + ",")) {
      outs() << "  = " << Cls << " already calibrated\n";
      continue;
    }
    std::string Exe = Work + "/cand_" + Cls + ".exe";
    std::string Log = Work + "/build_" + Cls + ".log";
    if (compile("-poseidon-apply-rewrites=M0_" + Idx, Exe, Log)) {
      outs() << "  ! " << Cls << " did not build; left uncalibrated (" << Log
             << ")\n";
      continue;
    }
    double Ms = timeOf(Exe);
    if (Ms <= 0) {
      outs() << "  ! " << Cls << " did not run; left uncalibrated\n";
      continue;
    }
    Rows += "wmma_inkernel_rel," + Cls + "," + num(Ms / BaseMs) + "\n";
    outs() << "  + " << Cls << ' ' << num(Ms, "%.4f") << " ms  rel "
           << num(Ms / BaseMs) << '\n';
  }
  if (Rows.empty()) {
    outs() << "[calibrate] in-kernel: nothing new to write\n";
    return;
  }
  // The merge is per family, so re-measured classes replace the whole family:
  // carry the classes this run did not touch across.
  for (StringRef Line : llvm::split(Content, '\n'))
    if (Line.starts_with("wmma_inkernel_rel,")) {
      StringRef Cls = Line.split(',').second.split(',').first;
      if (!StringRef(Rows).contains(("wmma_inkernel_rel," + Cls + ",").str()))
        Rows += (Line + "\n").str();
    }
  mergeRows(Content, Rows);
  writeFile(Csv, Content);
}

// The same order the compiler resolves in, so --check answers the question a
// compile would ask.
std::string findExisting(const Device &D, const std::string &Preferred) {
  if (sys::fs::exists(Preferred))
    return Preferred;
  std::string Prefix = "cm_" + D.Arch + "_";
  for (const std::string &Dir : {defaultOutDir(), shipped("cost_models")}) {
    std::error_code EC;
    for (sys::fs::directory_iterator It(Dir, EC), E; It != E && !EC;
         It.increment(EC)) {
      StringRef Name = sys::path::filename(It->path());
      if (Name.starts_with(Prefix) && Name.ends_with(".csv"))
        return It->path();
    }
  }
  return Preferred;
}

int checkModel(const std::string &Preferred, const Device &D) {
  std::string Csv = findExisting(D, Preferred);
  if (!sys::fs::exists(Csv)) {
    errs() << "poseidon-calibrate: no cost model for " << D.Arch << " in "
           << defaultOutDir() << " or " << shipped("cost_models")
           << "; measure this device with poseidon-calibrate\n";
    return 1;
  }
  std::string Content = readFile(Csv);
  int Bad = 0;
  StringRef Arch;
  for (StringRef Line : llvm::split(Content, '\n'))
    if (Line.consume_front("# native_arch="))
      Arch = Line.trim();
  outs() << "cost model  " << Csv << '\n';
  outs() << "device      " << D.Name << " (" << D.Arch << ", id " << D.Id
         << ")\n";
  if (Arch != D.Arch) {
    outs() << "native_arch " << (Arch.empty() ? "<absent>" : Arch)
           << "  MISMATCH: this device is " << D.Arch << '\n';
    ++Bad;
  } else {
    outs() << "native_arch " << Arch << "  ok\n";
  }
  std::string Platform = platformPathFor(Csv);
  if (StringRef(Arch).starts_with("sm_")) {
    bool Have = sys::fs::exists(Platform);
    outs() << (Have ? "have        " : "MISSING     ") << Platform
           << "  (Herbie algebraic-search platform; regenerate with "
              "--only herbie-platform)\n";
    if (!Have)
      ++Bad;
  }
  struct {
    const char *Family;
    const char *What;
  } Families[] = {{"fmul", "per-op throughputs"},
                  {"wmma_mma_m16n16k16", "tensor-core tile prices"},
                  {"ozaki_dispatch_rel", "Ozaki-II host dispatch"},
                  {"tcec_dispatch_rel", "error-corrected host dispatch"},
                  {"direct_dispatch_rel", "reduced-precision host dispatch"},
                  {"wmma_inkernel_rel", "in-kernel tensor-core raises"}};
  for (const auto &F : Families) {
    bool Have = hasFamily(Content, F.Family);
    outs() << (Have ? "have        " : "MISSING     ") << F.Family << "  ("
           << F.What << ")\n";
    if (!Have)
      ++Bad;
  }
  if (Bad)
    errs() << "poseidon-calibrate: " << Bad
           << " problem(s); re-run poseidon-calibrate --force on this device\n";
  return Bad ? 1 : 0;
}

} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(
      argc, argv,
      "measure this GPU's Poseidon cost model, or check an existing one\n");
  ToolDir = std::string(sys::path::parent_path(
      sys::fs::getMainExecutable(argv[0], (void *)(intptr_t)main)));

  // The platform arm reads a CSV and writes a file beside it. Asking for it
  // alone, against a named CSV, therefore needs no device present: that is how
  // a model measured on one machine gets its platform on another.
  if (Only == "herbie-platform" && StringRef(Out).ends_with(".csv")) {
    if (!sys::fs::exists(Out))
      fail("no cost model at " + Out);
    armHerbiePlatform(Out);
    return 0;
  }

  Device D = queryDevice();
  // Every child runs on the device that is being measured.
  ::setenv("CUDA_VISIBLE_DEVICES", D.Id.c_str(), 1);
  std::string Csv = resolveCsv(D);

  if (Check)
    return checkModel(Csv, D);

  outs() << "[calibrate] device " << D.Name << " (" << D.Arch << ", id " << D.Id
         << ")\n";
  if (wants("microbm"))
    armMicrobm(Csv);
  if (!sys::fs::exists(Csv))
    fail("no cost model at " + Csv +
         "; run the microbm arm before the dispatch arms");
  // Right after microbm, because it translates exactly what microbm wrote.
  if (wants("microbm") || wants("herbie-platform"))
    armHerbiePlatform(Csv);
  if (wants("ozaki"))
    armDispatch(Csv, "ozaki", "ozaki_dispatch_rel", "poseidon-ozaki-calibrate");
  if (wants("tcec"))
    armTcec(Csv, D);
  if (wants("direct"))
    armDispatch(Csv, "direct", "direct_dispatch_rel",
                "poseidon-direct-calibrate");
  if (wants("inkernel"))
    armInKernel(Csv, D);
  outs() << "[calibrate] cost model: " << Csv << '\n';
  return 0;
}
