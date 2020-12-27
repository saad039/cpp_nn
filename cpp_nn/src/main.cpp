#include<iostream>
#include<string>
#include<thread>
#include<type_traits>
#include<tensor.h>
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
    tensor<dtype,1024,1024> t;

    t.shape();

    t.range_fill(1);

    auto res = t.matmul(t);
    
    std::cout <<"Sum: "<<std::accumulate(res.begin(),res.end(),0.0f)<<std::endl;
    
}
