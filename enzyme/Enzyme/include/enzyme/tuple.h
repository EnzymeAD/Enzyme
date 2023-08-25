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

namespace enzyme {

template <int i>
struct Index {};

template <int i, typename T>
struct value_at_position { 
    T& operator[](Index<i>) { return value; }
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

template < int i, typename ... T>
auto & get(tuple< T ... > & tup) {
    return tup[Index<i>{}];
}

template < int i, typename ... T>
const auto & get(const tuple< T ... > & tup) {
    return tup[Index<i>{}];
}

}
