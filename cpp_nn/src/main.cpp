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





int main(int argc, char const *argv[])
{

    tensor_t<1024,1024> t;

    t.range_fill(1);
    
    const auto res = t.matmul(t);

    std::cout <<"Result: "<<std::reduce(std::execution::par_unseq,res.begin(),res.end(),0.0f) << std::endl;


    //model<3,3,3,3,3> m{std::move(t)};

    //m.summary();
    
}
