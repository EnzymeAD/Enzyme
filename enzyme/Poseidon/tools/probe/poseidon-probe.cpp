//===- poseidon-probe.cpp - measure a profiling run's condition numbers ---===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// poseidon-probe -- <program> [args...]
//
// Runs a profile-generation build of a program that calls poseidon_metric with
// POSEIDON_PROBE=1, so that at exit the profiler runtime re-executes the
// program to measure each site's condition number against that metric. The
// program replaces this process: its output, exit status and signals are its
// own.
//
//===----------------------------------------------------------------------===//

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

int main(int argc, char **argv) {
  int first = 1;
  if (first < argc && !strcmp(argv[first], "--"))
    ++first;
  if (first >= argc || argv[first][0] == '-') {
    fprintf(stderr,
            "usage: poseidon-probe -- <program> [args...]\n"
            "Runs a profile-generation build with POSEIDON_PROBE=1: at exit it "
            "runs itself again\nto measure each site's condition number "
            "against the metric it declares with\nposeidon_metric, and writes "
            "them into its profiles.\n");
    return 2;
  }
  setenv("POSEIDON_PROBE", "1", 1);
  execvp(argv[first], argv + first);
  fprintf(stderr, "poseidon-probe: cannot run %s: %s\n", argv[first],
          strerror(errno));
  return 127;
}
