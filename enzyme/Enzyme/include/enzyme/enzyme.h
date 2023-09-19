#pragma once

#include "tuple.h"
#include "type_traits.h"

#include <tuple>
#include <utility>
#include <type_traits>

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template < typename return_type, typename ... T >
return_type __enzyme_fwddiff(void*, T ... );

template < typename return_type, typename ... T >
return_type __enzyme_autodiff(void*, T ... );

namespace enzyme {
    
    enum ReturnActivity{
        INACTIVE,
        ACTIVE,
        DUPLICATED
    };

    struct nodiff{};

    template < typename T >
    struct active{
      T value;
      operator T&() { return value; }
    };

    template < typename T >
    active(T) -> active<T>;

    template < typename T >
    struct duplicated{  
      T value;
      T shadow;
    };

    template < typename T >
    struct inactive{
      T value;
    };

    template < typename T >
    struct type_info {
      static constexpr bool is_active = false; 

      #ifdef ENZYME_OMIT_INACTIVE
      using type = tuple<>;
      #else 
      using type = tuple<nodiff>;
      #endif
    };

    template < typename T >
    struct type_info < active<T> >{
        static constexpr bool is_active = true; 
        using type = tuple<T>;
    };

    template < typename ... T >
    struct concatenated;

    template < typename ... S, typename ... T, typename ... rest >
    struct concatenated < tuple < S ... >, tuple < T ... >, rest ... > {
        using type = typename concatenated< tuple< S ..., T ... >, rest ... >::type;
    };

    template < typename ... T >
    struct concatenated < tuple < T ... > > {
        using type = tuple< T ... >;
    };

    template < typename T >
    struct concatenated < tuple<T> > {
        using type = T;
    };

    // Yikes!
    // slightly cleaner in C++20, with std::remove_cvref
    template < typename ... T >
    struct autodiff_return {
        using type = typename concatenated< 
            typename type_info< 
                typename std::remove_reference<
                    typename std::remove_cv<
                        T
                    >::type
                >::type
            >::type ...
        >::type;
    };

    template < typename T >
    __attribute__((always_inline))
    auto expand_args(const enzyme::duplicated<T> & arg) {
        return enzyme::tuple<int, T, T>{enzyme_dup, arg.value, arg.shadow};
    }

    template < typename T >
    __attribute__((always_inline))
    auto expand_args(const enzyme::active<T> & arg) {
        return enzyme::tuple<int, T>{enzyme_out, arg.value};
    }

    template < typename T >
    __attribute__((always_inline))
    auto expand_args(const enzyme::inactive<T> & arg) {
        return enzyme::tuple<int, T>{enzyme_const, arg.value};
    }

    namespace detail {
        template <class F, class Tuple, std::size_t... I>
        __attribute__((always_inline))
        constexpr decltype(auto) apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
            return f(enzyme::get<I>(std::forward<Tuple>(t))...);
        }

        template <class F, class Tuple>
        __attribute__((always_inline))
        constexpr decltype(auto) apply(F&& f, Tuple&& t)
        {
            auto iseq = std::make_index_sequence<enzyme::tuple_size_v<std::remove_reference_t<Tuple>>>{};
            return apply_impl(std::forward<F>(f), std::forward<Tuple>(t), iseq);
        }

    }  // namespace detail 

    template < typename return_type, typename function, typename ... enz_arg_types >
    __attribute__((always_inline))
    auto autodiff_impl(function && f, enzyme::tuple< void*, enz_arg_types ... > && arg_tup) {
      return detail::apply(__enzyme_autodiff<return_type, enz_arg_types ... >, arg_tup);
    }

    template < typename function, typename ... arg_types>
    __attribute__((always_inline))
    auto autodiff(function && f, arg_types && ... args) {
        using return_type = typename autodiff_return<arg_types...>::type;
        return autodiff_impl<return_type>(f, enzyme::tuple_cat(enzyme::tuple{(void*)f}, expand_args(args)...));
    }

}