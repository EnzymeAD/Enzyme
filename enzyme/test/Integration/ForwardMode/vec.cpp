// RUN: %clang++ -std=c++11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme
// -enzyme -S | %lli - RUN: %clang++ -std=c++11 -O1 %s -S -emit-llvm -o - | %opt
// - %loadEnzyme -enzyme -S | %lli - RUN: %clang++ -std=c++11 -O2 %s -S
// -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - RUN: %clang++
// -std=c++11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme
// -enzyme -enzyme-inline=1 -S | %lli - RUN: %clang++ -std=c++11 -O1 %s -S
// -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme
// -enzyme -enzyme-inline=1 -S | %lli - RUN: %clang++ -std=c++11 -O3 %s -S
// -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -

#include <assert.h>
#include <vector>
#include <stdio.h>

double f(const std::vector<double> &control) {
  double result = 0;
  for (std::size_t i = 0; i < control.size(); i ++) {
    result += control[i] * control[i];
  }
  return result;
};

extern double __enzyme_fwddiff(double (*)(const std::vector<double> &),
                               std::vector<double> &, std::vector<double> &);

int main(int argc, char *argv[])
{
  std::vector<double> control{1.0, 1.0, 1.0, 1.0};
  std::vector<double> activity{0.0, 0.0, 0.0, 0.0};
  double dret = __enzyme_fwddiff(f, control, activity);
  printf("hello! dret: %f, %f, %f, %f, %f\n", dret, activity[0], activity[1], activity[2], activity[3]);

  return 0;
}

