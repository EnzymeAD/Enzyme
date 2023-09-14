#pragma once

#include "tuple.h"

#include <tuple> // for std::apply
#include <type_traits>

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template < typename return_type, typename ... T >
return_type __enzyme_fwddiff(void*, T ... );

#if 1
// getting undefined reference issues with 
// std::apply(__enzyme_autodiff<...>(...))
// when this func template is left undefined
template < typename return_type, typename ... T >
return_type __enzyme_autodiff(T ... );
#else
// getting wrong answers when providing a dummy impl
// in an attempt to work around undef ref issues
template < typename return_type, typename ... T >
return_type __enzyme_autodiff(T ... ) { return {}; }
#endif

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
    auto splat(const enzyme::duplicated<T> & arg) {
        return std::tuple<int, T, T>{enzyme_dup, arg.value, arg.shadow};
    }

    template < typename T >
    auto splat(const enzyme::active<T> & arg) {
        return std::tuple<int, T>{enzyme_out, arg.value};
    }

    template < typename T >
    auto splat(const enzyme::inactive<T> & arg) {
        return std::tuple<int, T>{enzyme_const, arg.value};
    }

    template < typename return_type, typename function, typename ... enz_arg_types >
    auto autodiff_impl(function && f, std::tuple< enz_arg_types ... > && arg_tup) {
      return std::apply(__enzyme_autodiff<return_type, enz_arg_types ... >, arg_tup);
    }

    template < typename function, typename ... arg_types>
    auto autodiff(function && f, arg_types && ... args) {
        using return_type = typename autodiff_return<arg_types...>::type;
        return autodiff_impl<return_type>(f, std::tuple_cat(std::tuple{f}, splat(args)...));
    }

}