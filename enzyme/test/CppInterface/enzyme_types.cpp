#include <utility>
#include <iostream>

#include <enzyme/enzyme>

struct move_only_type {
  move_only_type() {};
  move_only_type(move_only_type && obj) {
    std::cout << "calling move_only_type move ctor" << std::endl;
  };
};

int main() {
  double z = 3.0;
  double & rz = z;
  move_only_type foo{};

  [[maybe_unused]] enzyme::Const<double> c1{3.0};
  [[maybe_unused]] enzyme::Const<double> c2{z};
  [[maybe_unused]] enzyme::Const<double&> c3{rz};
  [[maybe_unused]] enzyme::Const<double&> c4{z};
  [[maybe_unused]] enzyme::Const<move_only_type> c5{move_only_type{}};
  [[maybe_unused]] enzyme::Const<move_only_type> c6{std::move(foo)};

// CTAD examples for C++17 and later
#if __cplusplus >= 201703L 
  [[maybe_unused]] enzyme::Const d1{3.0};
  [[maybe_unused]] enzyme::Const d2{z};
  [[maybe_unused]] enzyme::Const d3{rz};
#endif

}