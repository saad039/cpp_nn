#include <iostream>
#include <string>
#include <thread>
#include <type_traits>
#include <tensor.h>
#include <sys/resource.h>
#include <experimental/array>
#include "model.h"
#include <rng.h>

template<typename T, T... ints>
consteval auto array_iota(std::integer_sequence<T, ints...> int_seq)
{
    return std::array<T,int_seq.size()>{{ints...}};
}

int main(int argc, char const *argv[])
{

    constexpr tensor_t<4096,1> x; 
    constexpr tensor_t<4096,1> y; 

    constexpr auto dotproduct = x.dot(y);

    printf("%.8f\n",dotproduct);
}
