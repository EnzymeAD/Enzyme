#pragma once

#include "tuple.h"

namespace enzyme {

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
    struct inactive{};

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

    template < typename ... T >
    struct autodiff_return {
        using type = typename concatenated< 
            typename type_info<T>::type ...
        >::type;
    };

    template < typename function, typename ... arg_types >
    typename autodiff_return<arg_types...>::type autodiff(function && f, arg_types && ... args) {
      // TODO
    }

}
