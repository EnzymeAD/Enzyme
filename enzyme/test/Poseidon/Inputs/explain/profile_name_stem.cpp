// Prints, for each name on the command line, the stem of the profile file the
// FP profiler writes and the pass reads for a function of that name.
#include "runtime/fpprofiler/FPProfileName.h"

#include <cstdio>

int main(int argc, char **argv) {
  for (int I = 1; I < argc; ++I)
    std::puts(poseidon::profileNameStem(std::string(argv[I])).c_str());
  return 0;
}
