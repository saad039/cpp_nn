#include<iostream>
#include<string>
#include<thread>
#include<type_traits>
#include<tensor.h>
#include <sys/resource.h>
#include "model.h"
// template<typename T>
// concept floating_point  = std::is_floating_point_v<T>;

// template<floating_point dtype>
// auto sum(const dtype& a, const dtype& b) -> dtype
// {
//     return a + b;
// }

template <typename... Args>
auto sum_all(Args... args)  requires(... and std::is_floating_point_v<Args>) 
{ 
    return (... + args); 
}

using dtype = float;




int main(int argc, char const *argv[])
{

    tensor<dtype,3,3> t;

    t.range_fill(1);
    std::cout <<"Result: " << std::endl;
    t.matmul(t).print();
    
}
