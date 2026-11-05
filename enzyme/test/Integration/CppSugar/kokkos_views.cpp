// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

#include "../test_utils.h"

#include <enzyme/enzyme>

namespace Kokkos {

template <class DataType, class... Properties>
class View {
  using ElemT = std::remove_pointer_t<DataType>;
  void *tracker_;
  ElemT *ptr_;
  std::size_t extent_;

public:
  View() : tracker_(nullptr), ptr_(nullptr), extent_(0) {}
  View(ElemT *p, std::size_t n = 0) : tracker_(nullptr), ptr_(p), extent_(n) {}

  ElemT &operator()(int i) const { return ptr_[i]; }
  ElemT *data() const { return ptr_; }
  std::size_t extent(int) const { return extent_; }
};

namespace Experimental {
template <class DataType, class... Properties>
class View {
  using ElemT = std::remove_pointer_t<DataType>;
  void *tracker_;
  ElemT *ptr_;
  std::size_t extent_;

public:
  View() : tracker_(nullptr), ptr_(nullptr), extent_(0) {}
  View(ElemT *p, std::size_t n = 0) : tracker_(nullptr), ptr_(p), extent_(n) {}

  ElemT &operator()(int i) const { return ptr_[i]; }
};
} // namespace Experimental

template <class DataType, class... Properties>
class DualView {
public:
  View<DataType, Properties...> d_view;
  View<DataType, Properties...> h_view;
  std::size_t n;

  DualView() : d_view(), h_view(), n(0) {}
  DualView(std::remove_pointer_t<DataType> *dp,
           std::remove_pointer_t<DataType> *hp, std::size_t len)
      : d_view(dp), h_view(hp), n(len) {}
};

} // namespace Kokkos

struct SpanLike {
  using element_type = double;
  using pointer = double *;
  double *p;
  std::size_t n;
  double *data() const { return p; }
  std::size_t size() const { return n; }
};

struct NonView {
  double *p;
};

static_assert(enzyme::is_view_v<Kokkos::View<double *>>,
              "Kokkos::View must satisfy enzyme::is_view");
static_assert(enzyme::is_view_v<const Kokkos::View<double *>>,
              "const Kokkos::View must satisfy enzyme::is_view");
static_assert(enzyme::is_view_v<Kokkos::Experimental::View<double *>>,
              "Kokkos::Experimental::View must satisfy enzyme::is_view");
static_assert(enzyme::is_view_v<Kokkos::DualView<double *>>,
              "Kokkos::DualView must satisfy enzyme::is_view");
static_assert(enzyme::is_view_v<SpanLike>,
              "Span-like wrapper must satisfy enzyme::is_view");
static_assert(!enzyme::is_view_v<NonView>,
              "Plain structs must not satisfy enzyme::is_view");

double sum_squares_by_val(Kokkos::View<double *> x, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    sum += 2.5 * x(i) * x(i);
  return sum;
}

double experimental_and_dual_sum(Kokkos::Experimental::View<double *> ev,
                                 Kokkos::DualView<double *> dv) {
  double acc = 0.0;
  for (std::size_t i = 0; i < dv.n; ++i)
    acc += 3.0 * ev(i) + 5.0 * dv.d_view(i);
  return acc;
}

int main() {
  double x_buf[4] = {1.0, 2.0, 3.0, 4.0};
  double dx_buf[4] = {0.0, 0.0, 0.0, 0.0};
  Kokkos::View<double *> x(x_buf);
  Kokkos::View<double *> dx(dx_buf);

  enzyme::autodiff<enzyme::Reverse>(
      sum_squares_by_val, enzyme::Duplicated<Kokkos::View<double *>>{x, dx},
      enzyme::Const<int>{4});
  for (int i = 0; i < 4; ++i)
    APPROX_EQ(dx_buf[i], 5.0 * x_buf[i], 1e-10);

  double ev_buf[2] = {1.0, 2.0}, dev_buf[2] = {3.0, 4.0};
  double dv_buf[2] = {5.0, 6.0}, ddv_buf[2] = {0.5, 1.5};
  Kokkos::Experimental::View<double *> ev(ev_buf), dev(dev_buf);
  Kokkos::DualView<double *> dv(dv_buf, dv_buf, 2), ddv(ddv_buf, ddv_buf, 2);

  double fwd_res = enzyme::get<0>(enzyme::autodiff<enzyme::Forward>(
      experimental_and_dual_sum,
      enzyme::Duplicated<Kokkos::Experimental::View<double *>>{ev, dev},
      enzyme::Duplicated<Kokkos::DualView<double *>>{dv, ddv}));
  APPROX_EQ(fwd_res, 3.0 * (3.0 + 4.0) + 5.0 * (0.5 + 1.5), 1e-10);
  return 0;
}
