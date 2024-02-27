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

    struct nodiff{};

    template<bool ReturnPrimal = false>
    struct ReverseMode {

    };
    using Reverse = ReverseMode<false>;
    using ReverseWithPrimal = ReverseMode<true>;

    template < typename T >
    struct Active{
      T value;
      Active(T &&v) : value(v) {}
      operator T&() { return value; }
    };

    template < typename T >
    struct Duplicated{  
      T value;
      T shadow;
      Duplicated(T &&v, T&& s) : value(v), shadow(s) {}
    };

    template < typename T >
    struct Const{
      T value;
      Const(T &&v) : value(v) {}
      operator T&() { return value; }
    };

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

    template < typename T >
    __attribute__((always_inline))
    auto expand_args(const enzyme::Duplicated<T> & arg) {
        return enzyme::tuple<int, T, T>{enzyme_dup, arg.value, arg.shadow};
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

        template <class return_type, class Tuple, std::size_t... I>
        __attribute__((always_inline))
        constexpr decltype(auto) rev_apply_impl(void* f, int* ret_attr, Tuple&& t, std::index_sequence<I...>) {
            return push_return_last(__enzyme_autodiff<return_type>(f, ret_attr, enzyme::get<I>(impl::forward<Tuple>(t))...));
        }
        
        template <typename function, class Tuple, std::size_t... I>
        __attribute__((always_inline))
        constexpr decltype(auto) primal_apply_impl(function &&f, Tuple&& t, std::index_sequence<I...>) {
            return f(enzyme::get<I>(impl::forward<Tuple>(t))...);
        }

        template < typename T >
        struct default_ret_activity {
          using type = Const<T>;
        };
        
        template <>
        struct default_ret_activity<float> {
          using type = Active<float>;
        };
        
        template <>
        struct default_ret_activity<double> {
          using type = Active<double>;
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

    template < typename return_type, typename function, typename RetActivity, typename ... enz_arg_types >
    __attribute__((always_inline))
    auto rev_autodiff_impl(function && f, enzyme::tuple< enz_arg_types ... > && arg_tup) {
      using Tuple = enzyme::tuple< enz_arg_types ... >;
      return detail::rev_apply_impl<return_type>((void*)f, detail::ret_global<RetActivity>::value, impl::forward<Tuple>(arg_tup), std::make_index_sequence<enzyme::tuple_size_v<Tuple>>{});
    }
    
    template < typename DiffMode, typename RetActivity, typename function, typename ... arg_types>
    __attribute__((always_inline))
    auto autodiff(function && f, arg_types && ... args) {
        using return_type = typename autodiff_return<DiffMode, RetActivity, arg_types...>::type;
        return rev_autodiff_impl<return_type, function, RetActivity>(impl::forward<function>(f), enzyme::tuple_cat(enzyme::tuple{detail::ret_used<DiffMode, RetActivity>::value}, expand_args(args)...));
    }
 
    template < typename DiffMode, typename function, typename ... arg_types>
    __attribute__((always_inline))
    auto autodiff(function && f, arg_types && ... args) {
        using primal_return_type = decltype(primal_call<function, arg_types...>(impl::forward<function>(f), impl::forward<arg_types>(args)...));
        using RetActivity = typename detail::default_ret_activity<primal_return_type>::type;
        using return_type = typename autodiff_return<DiffMode, RetActivity, arg_types...>::type;
        return rev_autodiff_impl<return_type, function, RetActivity>(impl::forward<function>(f), enzyme::tuple_cat(enzyme::tuple{detail::ret_used<DiffMode, RetActivity>::value}, expand_args(args)...));
    }
}
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

#include <utility> // for std::integer_sequence

#include <enzyme/type_traits>

#define _NOEXCEPT noexcept
namespace enzyme {

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
struct tuple_size<tuple<T...>> : std::integral_constant<size_t, sizeof...(T)> {};

template <typename Tuple>
static constexpr size_t tuple_size_v = tuple_size<Tuple>::value;

template <typename... T>
__attribute__((always_inline))
constexpr auto forward_as_tuple(T&&... args) noexcept {
  return tuple<T&&...>{impl::forward<T>(args)...};
}

namespace impl {

template <typename index_seq>
struct make_tuple_from_fwd_tuple;

template <size_t... indices>
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

template <size_t... fwd_indices, size_t... indices>
struct concat_with_fwd_tuple<std::index_sequence<fwd_indices...>, std::index_sequence<indices...>> {
  template <typename FWD_TUPLE, typename TUPLE>
  __attribute__((always_inline))
  static constexpr auto f(FWD_TUPLE&& fwd, TUPLE&& t) {
    return forward_as_tuple(get<fwd_indices>(impl::forward<FWD_TUPLE>(fwd))..., get<indices>(impl::forward<TUPLE>(t))...);
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
