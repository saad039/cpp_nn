#include <iostream>
#include <string>
#include <thread>
#include <type_traits>

template <typename... Args>
auto sum_all(Args... args) requires(... and std::is_floating_point_v<Args>)
{
    return (... + args);
}

using dtype = double;

int main(int argc, char const *argv[])
{
    return 0;
}
