#include <array>
#include <iostream>
#include <rng.h>
#include <string>
#include <sys/resource.h>
#include <tensor.h>
#include <thread>
#include <type_traits>

template <typename T, T... ints>
consteval auto array_iota(std::integer_sequence<T, ints...> int_seq) {
    return std::array<T, int_seq.size()>{{ints...}};
}

int main(int, char const*[]) {

    constexpr tensor_t<4096, 1> x;
    constexpr tensor_t<4096, 1> y;

    constexpr auto dotproduct = x.dot(y);

    printf("%.8f\n", dotproduct);
}
