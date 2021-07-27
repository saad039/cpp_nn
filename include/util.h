#if !defined(UTIL_H)
#define UTIL_H

#include <cstdint>
#include <tuple>
#include <type_traits>

namespace util {
auto println = [](auto&&... args) {
    (std::cout << ... << args);
    std::cout << std::endl;
};

auto printshape = [](const char* str, const std::pair<std::size_t, std::size_t>& shape) {
    println(str, "(", shape.first, ",", shape.second, ")");
};

namespace helper {
template <typename T>
concept floating_point = std::is_floating_point_v<T>;
template <floating_point dtype>
constexpr double sqrtNewtonRaphson(const dtype x, const dtype curr, const dtype prev) {
    return curr == prev ? curr
                        : sqrtNewtonRaphson(x, static_cast<dtype>(0.5) * (curr + x / curr), curr);
}
} // namespace helper

template <typename T>
concept floating_point = std::is_floating_point_v<T>;
template <floating_point dtype>

constexpr double sqrt(const dtype x) {
    return x >= 0 && x < std::numeric_limits<dtype>::infinity()
               ? helper::sqrtNewtonRaphson(x, x, static_cast<dtype>(0))
               : std::numeric_limits<dtype>::quiet_NaN();
}

template <typename T, T... ints>
constexpr auto array_iota(std::integer_sequence<T, ints...> int_seq) {
    return std::array<T, int_seq.size()>{{ints...}};
}

template <typename T>
constexpr auto relu(const T& x) {
    return std::max(x, static_cast<T>(0));
}

template <typename T>
constexpr auto sigmoid(const T& x) {
    return static_cast<T>(1.0 / (1 + exp(-x)));
}

} // namespace util

#endif // UTIL_H
