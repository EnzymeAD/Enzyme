
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

}