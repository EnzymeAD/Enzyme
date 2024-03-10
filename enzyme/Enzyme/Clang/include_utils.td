class Headers<string filename_, string contents_> {
  string filename = filename_;
  string contents = contents_;
}

def : Headers<"/enzymeroot/enzyme/utils", [{
#pragma once

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;

extern int enzyme_const_return;
extern int enzyme_active_return;
extern int enzyme_dup_return;

extern int enzyme_primal_return;
extern int enzyme_noret;

template<typename Return, typename... T>
Return __enzyme_autodiff(T...);

template<typename Return, typename... T>
Return __enzyme_fwddiff(T...);

#include <enzyme/tuple>

namespace enzyme {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"

    struct nodiff{};

    template<bool ReturnPrimal = false>
    struct ReverseMode {

    };
    using Reverse = ReverseMode<false>;
    using ReverseWithPrimal = ReverseMode<true>;
    
    struct ForwardMode {

    };
    using Forward = ForwardMode;

    template < typename T >
    struct Active{
      T value;
      operator T&() { return value; }
    };

    template < typename T >
    struct Duplicated{  
      T value;
      T shadow;
    };

    template < typename T >
    struct DuplicatedNoNeed{  
      T value;
      T shadow;
    };

    template < typename T >
    struct Const{
      T value;
      operator T&() { return value; }
    };

    // CTAD available in C++17 or later
    #if __cplusplus >= 201703L 
      template < typename T >
      Active(T) -> Active<T>;

      template < typename T >
      Const(T) -> Const<T>;

      template < typename T >
      Duplicated(T,T) -> Duplicated<T>;

      template < typename T >
      DuplicatedNoNeed(T,T) -> DuplicatedNoNeed<T>;
    #endif    

    template < typename T >
    struct type_info {
      static constexpr bool is_active = false; 
      using type = nodiff;
    };

    template < typename T >
    struct type_info < Active<T> >{
        static constexpr bool is_active = true; 
        using type = T;
    };

    template < typename ... T >
    struct concatenated;

    template < typename ... S, typename T, typename ... rest >
    struct concatenated < tuple < S ... >, T, rest ... > {
        using type = typename concatenated< tuple< S ..., T>, rest ... >::type;
    };

    template < typename T >
    struct concatenated < T > {
        using type = T;
    };

    // Yikes!
    // slightly cleaner in C++20, with std::remove_cvref
    template < typename ... T >
    struct autodiff_return;
    
    template < typename RetType, typename ... T >
    struct autodiff_return<ReverseMode<false>, RetType, T...>
    {
        using type = tuple<typename concatenated< tuple< >,  
            typename type_info< 
                typename remove_cvref< T >::type
            >::type ...
        >::type>;
    };
    
    template < typename RetType, typename ... T >
    struct autodiff_return<ReverseMode<true>, RetType, T...>
    {
        using type = tuple<
            typename type_info<RetType>::type,
            typename concatenated< tuple< >,  
                typename type_info< 
                    typename remove_cvref< T >::type
                >::type ...
            >::type
        >;
    };
    
    template < typename T0, typename ... T >
    struct autodiff_return<ForwardMode, Const<T0>, T...>
    {
        using type = tuple<T0>;
    };
    
    template < typename T0, typename ... T >
    struct autodiff_return<ForwardMode, Duplicated<T0>, T...>
    {
        using type = tuple<T0, T0>;
    };
    
    template < typename T0, typename ... T >
    struct autodiff_return<ForwardMode, DuplicatedNoNeed<T0>, T...>
    {
        using type = tuple<T0>;
    };

    template < typename T >
    __attribute__((always_inline))
    auto expand_args(const enzyme::Duplicated<T> & arg) {
        return enzyme::tuple<int, T, T>{enzyme_dup, arg.value, arg.shadow};
    }

    template < typename T >
    __attribute__((always_inline))
    auto expand_args(const enzyme::DuplicatedNoNeed<T> & arg) {
        return enzyme::tuple<int, T, T>{enzyme_dupnoneed, arg.value, arg.shadow};
    }

    template < typename T >
    __attribute__((always_inline))
    auto expand_args(const enzyme::Active<T> & arg) {
        return enzyme::tuple<int, T>{enzyme_out, arg.value};
    }

    template < typename T >
    __attribute__((always_inline))
    auto expand_args(const enzyme::Const<T> & arg) {
        return enzyme::tuple<int, T>{enzyme_const, arg.value};
    }

    template < typename T >
    __attribute__((always_inline))
    auto primal_args(const enzyme::Duplicated<T> & arg) {
        return enzyme::tuple<T>{arg.value};
    }

    template < typename T >
    __attribute__((always_inline))
    auto primal_args(const enzyme::DuplicatedNoNeed<T> & arg) {
        return enzyme::tuple<T>{arg.value};
    }

    template < typename T >
    __attribute__((always_inline))
    auto primal_args(const enzyme::Active<T> & arg) {
        return enzyme::tuple<T>{arg.value};
    }

    template < typename T >
    __attribute__((always_inline))
    auto primal_args(const enzyme::Const<T> & arg) {
        return enzyme::tuple<T>{arg.value};
    }

    namespace detail {
        template<typename T>
        __attribute__((always_inline))
        constexpr decltype(auto) push_return_last(T &&t);
        
        template<typename ...T>
        __attribute__((always_inline))
        constexpr decltype(auto) push_return_last(tuple<tuple<T...>> &&t) {
          return tuple<tuple<T...>>{get<0>(t)};
        }
        
        template<typename ...T, typename R>
        __attribute__((always_inline))
        constexpr decltype(auto) push_return_last(tuple<R, tuple<T...>> &&t) {
          return tuple{get<1>(t), get<0>(t)};
        }

        template <typename Mode>
        struct autodiff_apply {};

        template <bool Mode>
        struct autodiff_apply<ReverseMode<Mode>> {
        template <class return_type, class Tuple, std::size_t... I>
        __attribute__((always_inline))
        static constexpr decltype(auto) impl(void* f, int* ret_attr, Tuple&& t, std::index_sequence<I...>) {
            return push_return_last(__enzyme_autodiff<return_type>(f, ret_attr, enzyme::get<I>(impl::forward<Tuple>(t))...));
        }
        };

        template <>
        struct autodiff_apply<ForwardMode> {
        template <class return_type, class Tuple, std::size_t... I>
        __attribute__((always_inline))
        static constexpr return_type impl(void* f, int* ret_attr, Tuple&& t, std::index_sequence<I...>) {
            return __enzyme_fwddiff<return_type>(f, ret_attr, enzyme::get<I>(impl::forward<Tuple>(t))...);
        }
        };
        
        template <typename function, class Tuple, std::size_t... I>
        __attribute__((always_inline))
        constexpr decltype(auto) primal_apply_impl(function &&f, Tuple&& t, std::index_sequence<I...>) {
            return f(enzyme::get<I>(impl::forward<Tuple>(t))...);
        }

        template < typename Mode, typename T >
        struct default_ret_activity {
          using type = Const<T>;
        };
        
        template <bool prim>
        struct default_ret_activity<ReverseMode<prim>, float> {
          using type = Active<float>;
        };
        
        template <bool prim>
        struct default_ret_activity<ReverseMode<prim>, double> {
          using type = Active<double>;
        };
       
        template<>
        struct default_ret_activity<ForwardMode, float> {
          using type = DuplicatedNoNeed<float>;
        };
        
        template<>
        struct default_ret_activity<ForwardMode, double> {
          using type = DuplicatedNoNeed<double>;
        };

        template < typename T >
        struct ret_global;
        
        template<typename T>
        struct ret_global<Const<T>> {
          static constexpr int* value = &enzyme_const_return;
        };
        
        template<typename T>
        struct ret_global<Active<T>> {
          static constexpr int* value = &enzyme_active_return;
        };
        
        template<typename T>
        struct ret_global<Duplicated<T>> {
          static constexpr int* value = &enzyme_dup_return;
        };
        
        template<typename T>
        struct ret_global<DuplicatedNoNeed<T>> {
          static constexpr int* value = &enzyme_dup_return;
        };
        
        template<typename Mode, typename RetAct>
        struct ret_used;

        template<typename RetAct>
        struct ret_used<ReverseMode<true>, RetAct> {
          static constexpr int* value = &enzyme_primal_return;
        };

        template<typename RetAct>
        struct ret_used<ReverseMode<false>, RetAct> {
          static constexpr int* value = &enzyme_noret;
        };

        template<typename T>
        struct ret_used<ForwardMode, DuplicatedNoNeed<T>> {
          static constexpr int* value = &enzyme_noret;
        };
        template<typename T>
        struct ret_used<ForwardMode, Const<T>> {
          static constexpr int* value = &enzyme_primal_return;
        };
        template<typename T>
        struct ret_used<ForwardMode, Duplicated<T>> {
          static constexpr int* value = &enzyme_primal_return;
        };

    }  // namespace detail 

    template < typename return_type, typename function, typename ... enz_arg_types >
    __attribute__((always_inline))
    auto primal_impl(function && f, enzyme::tuple< enz_arg_types ... > && arg_tup) {
      using Tuple = enzyme::tuple< enz_arg_types ... >;
      return detail::primal_apply_impl<return_type>(f, impl::forward<Tuple>(arg_tup), std::make_index_sequence<enzyme::tuple_size_v<Tuple>>{});
    }

    template < typename function, typename ... arg_types>
    auto primal_call(function && f, arg_types && ... args) {
        return primal_impl<function>(impl::forward<function>(f), enzyme::tuple_cat(primal_args(args)...));
    }

    template < typename return_type, typename DiffMode, typename function, typename RetActivity, typename ... enz_arg_types >
    __attribute__((always_inline))
    auto autodiff_impl(function && f, enzyme::tuple< enz_arg_types ... > && arg_tup) {
      using Tuple = enzyme::tuple< enz_arg_types ... >;
      return detail::autodiff_apply<DiffMode>::template impl<return_type>((void*)f, detail::ret_global<RetActivity>::value, impl::forward<Tuple>(arg_tup), std::make_index_sequence<enzyme::tuple_size_v<Tuple>>{});
    }
    
    template < typename DiffMode, typename RetActivity, typename function, typename ... arg_types>
    __attribute__((always_inline))
    auto autodiff(function && f, arg_types && ... args) {
        using return_type = typename autodiff_return<DiffMode, RetActivity, arg_types...>::type;
        return autodiff_impl<return_type, DiffMode, function, RetActivity>(impl::forward<function>(f), enzyme::tuple_cat(enzyme::tuple{detail::ret_used<DiffMode, RetActivity>::value}, expand_args(args)...));
    }
 
    template < typename DiffMode, typename function, typename ... arg_types>
    __attribute__((always_inline))
    auto autodiff(function && f, arg_types && ... args) {
        using primal_return_type = decltype(primal_call<function, arg_types...>(impl::forward<function>(f), impl::forward<arg_types>(args)...));
        using RetActivity = typename detail::default_ret_activity<DiffMode, primal_return_type>::type;
        using return_type = typename autodiff_return<DiffMode, RetActivity, arg_types...>::type;
        return autodiff_impl<return_type, DiffMode, function, RetActivity>(impl::forward<function>(f), enzyme::tuple_cat(enzyme::tuple{detail::ret_used<DiffMode, RetActivity>::value}, expand_args(args)...));
    }
#pragma clang diagnostic pop

} // namespace enzyme
}]>;

def : Headers<"/enzymeroot/enzyme/type_traits", [{
#pragma once

#include <type_traits>

namespace enzyme {

// this is already in C++20, but we reimplement it here for older C++ versions
template < typename T >
struct remove_cvref {
    using type = 
        typename std::remove_reference<
            typename std::remove_cv<
                T
            >::type
        >::type;
};

template < typename T >
using remove_cvref_t = typename remove_cvref<T>::type;

namespace impl {
  template<typename _Tp>
    __attribute__((always_inline))
    constexpr _Tp&&
    forward(std::remove_reference_t<_Tp>& __t) noexcept
    { return static_cast<_Tp&&>(__t); }

  /**
   *  @brief  Forward an rvalue.
   *  @return The parameter cast to the specified type.
   *
   *  This function is used to implement "perfect forwarding".
   */
  template<typename _Tp>
    __attribute__((always_inline))
    constexpr _Tp&&
    forward(std::remove_reference_t<_Tp>&& __t) noexcept
    {
      static_assert(!std::is_lvalue_reference<_Tp>::value,
	  "enzyme::impl::forward must not be used to convert an rvalue to an lvalue");
      return static_cast<_Tp&&>(__t);
    }

}

}
}]>;

def : Headers<"/enzymeroot/enzyme/tuple", [{
#pragma once

/////////////
// tuple.h //
/////////////

// why reinvent the wheel and implement a tuple class?
//  - ensure data is laid out in the same order the types are specified
//        see: https://github.com/EnzymeAD/Enzyme/issues/1191#issuecomment-1556239213
//  - CUDA compatibility: std::tuple has some compatibility issues when used
//        in a __device__ context (this may get better in c++20 with the improved
//        constexpr support for std::tuple). Owning the implementation lets
//        us add __host__ __device__ annotations to any part of it

#include <cstddef> // for std::size_t
#include <utility> // for std::integer_sequence

#include <enzyme/type_traits>

#define _NOEXCEPT noexcept
namespace enzyme {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"

template <int i>
struct Index {};

template <int i, typename T>
struct value_at_position { 
  __attribute__((always_inline))
  T & operator[](Index<i>) { return value; }

  __attribute__((always_inline))
  constexpr const T & operator[](Index<i>) const { return value; }
  T value;
};

template <typename S, typename... T>
struct tuple_base;

template <int... i, typename... T>
struct tuple_base<std::integer_sequence<int, i...>, T...>
    : public value_at_position<i, T>... {
    using value_at_position<i, T>::operator[]...;
}; 

template <typename... T>
struct tuple : public tuple_base<std::make_integer_sequence<int, sizeof...(T)>, T...> {};

template <typename... T>
__attribute__((always_inline))
tuple(T ...) -> tuple<T...>;

template < int i, typename Tuple >
__attribute__((always_inline))
decltype(auto) get(Tuple && tup) {
  constexpr bool is_lvalue = std::is_lvalue_reference_v<Tuple>;
  constexpr bool is_const = std::is_const_v<std::remove_reference_t<Tuple>>;
  using T = remove_cvref_t< decltype(tup[Index<i>{ } ]) >;
  if constexpr ( is_lvalue &&  is_const) { return static_cast<const T&>(tup[Index<i>{} ]); }
  if constexpr ( is_lvalue && !is_const) { return static_cast<T&>(tup[Index<i>{} ]); }
  if constexpr (!is_lvalue &&  is_const) { return static_cast<const T&&>(tup[Index<i>{} ]); }
  if constexpr (!is_lvalue && !is_const) { return static_cast<T&&>(tup[Index<i>{} ]); }
}

template < int i, typename ... T>
__attribute__((always_inline))
decltype(auto) get(const tuple< T ... > & tup) {
    return tup[Index<i>{} ];
}

template <typename Tuple>
struct tuple_size;

template <typename... T>
struct tuple_size<tuple<T...>> : std::integral_constant<std::size_t, sizeof...(T)> {};

template <typename Tuple>
static constexpr std::size_t tuple_size_v = tuple_size<Tuple>::value;

template <typename... T>
__attribute__((always_inline))
constexpr auto forward_as_tuple(T&&... args) noexcept {
  return tuple<T&&...>{impl::forward<T>(args)...};
}

namespace impl {

template <typename index_seq>
struct make_tuple_from_fwd_tuple;

template <std::size_t... indices>
struct make_tuple_from_fwd_tuple<std::index_sequence<indices...>> {
  template <typename FWD_TUPLE>
  __attribute__((always_inline))
  static constexpr auto f(FWD_TUPLE&& fwd) {
    return tuple{get<indices>(impl::forward<FWD_TUPLE>(fwd))...};
  }
};

template <typename FWD_INDEX_SEQ, typename TUPLE_INDEX_SEQ>
struct concat_with_fwd_tuple;

template < typename Tuple >
using iseq = std::make_index_sequence<tuple_size_v< enzyme::remove_cvref_t< Tuple > > >;

template <std::size_t... fwd_indices, std::size_t... indices>
struct concat_with_fwd_tuple<std::index_sequence<fwd_indices...>, std::index_sequence<indices...>> {
  template <typename FWD_TUPLE, typename TUPLE>
  __attribute__((always_inline))
  static constexpr auto f(FWD_TUPLE&& fwd, TUPLE&& t) {
    return enzyme::forward_as_tuple(get<fwd_indices>(impl::forward<FWD_TUPLE>(fwd))..., get<indices>(impl::forward<TUPLE>(t))...);
  }
};

template <typename Tuple>
__attribute__((always_inline))
static constexpr auto tuple_cat(Tuple&& ret) {
  return make_tuple_from_fwd_tuple< iseq< Tuple > >::f(impl::forward< Tuple >(ret));
}

template <typename FWD_TUPLE, typename first, typename... rest>
__attribute__((always_inline))
static constexpr auto tuple_cat(FWD_TUPLE&& fwd, first&& t, rest&&... ts) {
  return tuple_cat(concat_with_fwd_tuple< iseq<FWD_TUPLE>, iseq<first> >::f(impl::forward<FWD_TUPLE>(fwd), impl::forward<first>(t)), impl::forward<rest>(ts)...);
}

}  // namespace impl

template <typename... Tuples>
__attribute__((always_inline))
constexpr auto tuple_cat(Tuples&&... tuples) {
  return impl::tuple_cat(impl::forward<Tuples>(tuples)...);
}

#pragma clang diagnostic pop

} // namespace enzyme
#undef _NOEXCEPT
}]>;

def : Headers<"/enzymeroot/enzyme/enzyme", [{
#ifdef __cplusplus
#include "enzyme/utils"
#else
#warning "Enzyme wrapper templates only available in C++"
#endif
}]>;

def : Headers<"/enzymeroot/enzyme/mpfr", [{
//===- EnzymeMPFR.h - MPFR wrappers ---------------------------------------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains easy to use wrappers around MPFR functions.
//
//===----------------------------------------------------------------------===//
#ifndef __ENZYME_RUNTIME_ENZYME_MPFR__
#define __ENZYME_RUNTIME_ENZYME_MPFR__

#include <mpfr.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// TODO s
//
// (for MPFR ver. 2.1)
//
// We need to set the range of the allowed exponent using `mpfr_set_emin` and
// `mpfr_set_emax`. (This means we can also play with whether the range is
// centered around 0 (1?) or somewhere else)
//
// (also these need to be mutex'ed as the exponent change is global in mpfr and
// not float-specific) ... (mpfr seems to have thread safe mode - check if it is
// enabled or if it is enabled by default)
//
// For that we need to do this check:
//   If the user changes the exponent range, it is her/his responsibility to
//   check that all current floating-point variables are in the new allowed
//   range (for example using mpfr_check_range), otherwise the subsequent
//   behavior will be undefined, in the sense of the ISO C standard.
//
// MPFR docs state the following:
//   Note: Overflow handling is still experimental and currently implemented
//   partially. If an overflow occurs internally at the wrong place, anything
//   can happen (crash, wrong results, etc).
//
// Which we would like to avoid somehow.
//
// MPFR also has this limitation that we need to address for accurate
// simulation:
//   [...] subnormal numbers are not implemented.
//

#define __ENZYME_MPFR_SINGOP(OP_TYPE, LLVM_OP_NAME, MPFR_FUNC_NAME, FROM_TYPE, \
                             RET, MPFR_GET, ARG1, MPFR_SET_ARG1,               \
                             ROUNDING_MODE)                                    \
  __attribute__((weak))                                                        \
  RET __enzyme_mpfr_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME(                  \
      ARG1 a, int64_t exponent, int64_t significand) {                         \
    mpfr_t ma, mc;                                                             \
    mpfr_init2(ma, significand);                                               \
    mpfr_init2(mc, significand);                                               \
    mpfr_set_##MPFR_SET_ARG1(ma, a, ROUNDING_MODE);                            \
    mpfr_##MPFR_FUNC_NAME(mc, ma, ROUNDING_MODE);                              \
    RET c = mpfr_get_##MPFR_GET(mc, ROUNDING_MODE);                            \
    mpfr_clear(ma);                                                            \
    mpfr_clear(mc);                                                            \
    return c;                                                                  \
  }

#define __ENZYME_MPFR_BINOP(OP_TYPE, LLVM_OP_NAME, MPFR_FUNC_NAME, FROM_TYPE,  \
                            RET, MPFR_GET, ARG1, MPFR_SET_ARG1, ARG2,          \
                            MPFR_SET_ARG2, ROUNDING_MODE)                      \
  __attribute__((weak))                                                        \
  RET __enzyme_mpfr_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME(                  \
      ARG1 a, ARG2 b, int64_t exponent, int64_t significand) {                 \
    mpfr_t ma, mb, mc;                                                         \
    mpfr_init2(ma, significand);                                               \
    mpfr_init2(mb, significand);                                               \
    mpfr_init2(mc, significand);                                               \
    mpfr_set_##MPFR_SET_ARG1(ma, a, ROUNDING_MODE);                            \
    mpfr_set_##MPFR_SET_ARG1(mb, b, ROUNDING_MODE);                            \
    mpfr_##MPFR_FUNC_NAME(mc, ma, mb, ROUNDING_MODE);                          \
    RET c = mpfr_get_##MPFR_GET(mc, ROUNDING_MODE);                            \
    mpfr_clear(ma);                                                            \
    mpfr_clear(mb);                                                            \
    mpfr_clear(mc);                                                            \
    return c;                                                                  \
  }

#define __ENZYME_MPFR_DEFAULT_ROUNDING_MODE GMP_RNDN
#define __ENZYME_MPFR_DOUBLE_BINOP(LLVM_OP_NAME, MPFR_FUNC_NAME,               \
                                   ROUNDING_MODE)                              \
  __ENZYME_MPFR_BINOP(binop, LLVM_OP_NAME, MPFR_FUNC_NAME, 64_52, double, d,   \
                      double, d, double, d, ROUNDING_MODE)
#define __ENZYME_MPFR_DOUBLE_BINOP_DEFAULT_ROUNDING(LLVM_OP_NAME,              \
                                                    MPFR_FUNC_NAME)            \
  __ENZYME_MPFR_DOUBLE_BINOP(LLVM_OP_NAME, MPFR_FUNC_NAME,                     \
                             __ENZYME_MPFR_DEFAULT_ROUNDING_MODE)

__ENZYME_MPFR_DOUBLE_BINOP_DEFAULT_ROUNDING(fmul, mul)
__ENZYME_MPFR_DOUBLE_BINOP_DEFAULT_ROUNDING(fadd, add)
__ENZYME_MPFR_DOUBLE_BINOP_DEFAULT_ROUNDING(fdiv, div)

__ENZYME_MPFR_SINGOP(func, sqrt, sqrt, 64_52, double, d, double, d,
                     __ENZYME_MPFR_DEFAULT_ROUNDING_MODE)

#ifdef __cplusplus
}
#endif

#endif // #ifndef __ENZYME_RUNTIME_ENZYME_MPFR__
}]>;
