// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang++ -O1 %s %poseidon_profile_lib -o %t/meet
// RUN: env POSEIDON_PROFILE_DIR=%t/first MEET=device-first %t/meet | FileCheck --check-prefix=DEVICE-FIRST %s
// RUN: diff %t/first.expected %t/first/k_ran.fpprofile
// RUN: FileCheck --check-prefix=NEVER %s < %t/first/k_never.fpprofile
// RUN: env POSEIDON_PROFILE_DIR=%t/last MEET=device-last %t/meet | FileCheck --check-prefix=DEVICE-LAST %s
// RUN: diff %t/last.expected %t/last/k_ran.fpprofile
// RUN: FileCheck --check-prefix=NEVER %s < %t/last/k_never.fpprofile

// The two exit writers of a profile-generate CUDA or HIP program, without a
// GPU: the device profiler's is played by deviceWriter, which writes what it
// writes for a kernel that ran and reports it as the device profiler does. Both
// kernels' headers are registered with the host profiler, as the host half of
// their unit registers them. Whichever exit handler runs first, the kernel that
// ran keeps exactly the device writer's profile and the one that never ran
// gets its header and `Executed = 0`.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

extern "C" void poseidonRegisterProfileStatic(const char *funcName,
                                              const char *text);
extern "C" void poseidonProfileWritten(const char *funcName);

static const char kRan[] = "MaxBlockDims = 256 1 1\n"
                           "MaxGridDims = 4 1 1\n"
                           "LaunchCount = 1\n"
                           "SiteId = 0\n"
                           "CanonicalHash = 00000000000000aa\n"
                           "0\n"
                           "\tMinRes = 1.0\n"
                           "\tMaxRes = 2.0\n"
                           "\tSumValue = 3.0\n"
                           "\tSumSens = 3.0\n"
                           "\tSumGrad = 2.0\n"
                           "\tSumAbsGrad = 2.0\n"
                           "\tExec = 2\n"
                           "\tNumOperands = 0\n";

static void write(const std::string &path, const char *text) {
  FILE *f = fopen(path.c_str(), "w");
  if (!f)
    abort();
  fputs(text, f);
  fclose(f);
}

static void deviceWriter() {
  std::string dir = getenv("POSEIDON_PROFILE_DIR");
  struct stat st;
  printf("k_ran %s\n", stat((dir + "/k_ran.fpprofile").c_str(), &st) == 0
                           ? "written by the host profiler first"
                           : "not written yet");
  fflush(stdout);
  mkdir(dir.c_str(), 0755);
  write(dir + "/k_ran.fpprofile", kRan);
  write(dir + ".expected", kRan);
  poseidonProfileWritten("k_ran");
}

static bool deviceLast() {
  const char *m = getenv("MEET");
  return m && !strcmp(m, "device-last");
}

// Ahead of the host profiler's own static initializer, which registers its
// exit handler, as the registration constructors the host half emits are: an
// exit handler registered here runs after that one.
__attribute__((constructor(101))) static void early() {
  poseidonRegisterProfileStatic("k_ran",
                                "SiteId = 0\nCanonicalHash = 00000000000000aa\n");
  poseidonRegisterProfileStatic("k_never",
                                "SiteId = 1\nCanonicalHash = 00000000000000bb\n");
  if (deviceLast())
    atexit(deviceWriter);
}

int main() {
  // As the device profiler registers its exit handler, from main.
  if (!deviceLast())
    atexit(deviceWriter);
}

// DEVICE-FIRST: k_ran not written yet
// DEVICE-LAST: k_ran written by the host profiler first

// NEVER: SiteId = 1
// NEVER-NEXT: CanonicalHash = 00000000000000bb
// NEVER-NEXT: Executed = 0
// NEVER-NOT: {{.}}
