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

using dtype = double;

int main(int argc, char const *argv[])
{
    tensor<float,3,3> t;

    //t.shape();

    t.range_fill(1);
    
    t.print();
    std::cout <<"====================\n";

    t.matmul(t).print();


    //t.transpose();
    //std::cout <<t.dot(t)<<std::endl;
    
}
