#include <iostream>
#include <string>
#include <thread>
#include <type_traits>
#include <tensor.h>
#include <sys/resource.h>
#include <experimental/array>
#include "model.h"
#include <rng.h>
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



template<typename T, T... ints>
consteval auto array_iota(std::integer_sequence<T, ints...> int_seq)
{
    return std::array<T,int_seq.size()>{{ints...}};
}

#define LRATE 0.001f

int main(int argc, char const *argv[])
{

    


    constexpr tensor_t<4,2> x {{0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0}};
    constexpr tensor_t<4,1> y {{0.0,1.0,1.0,0.0}};

    tensor_t<2,2> w1{};
    tensor_t<1,2> b1{};

    tensor_t<2,1> w2{};
    tensor_t<1,1> b2{};


    //x.print();


    for(int i= 0; i < 5000; i++) {



//Forward propagation
        const auto hidden_layer_activation = x.matmul(w1) + b1;
        const auto hidden_layer_output = hidden_layer_activation._transform([](const auto x) {
             return 1.0f/(1.0f+exp(-x));
        });
        const auto output_layer_activation = hidden_layer_output.matmul(w2) + b2;
        const auto predicted_output = output_layer_activation._transform([](const auto x) {
             return 1.0f/(1.0f+exp(-x));
        }); //yhat
//Loss
        const auto error = y - predicted_output;
       
//Back Prop
        const auto d_predicted_output = error.element_wise_mul(predicted_output._transform([](const auto x){return x * (1.0f - x);}));
        const auto error_hidden_layer = d_predicted_output.matmul(w2.transpose());
 
        const auto d_hidden_layer = error_hidden_layer.element_wise_mul(hidden_layer_output._transform([](const auto x){return x * (1.0f - x);}));

        w2 = w2 + hidden_layer_output.transpose().matmul(d_predicted_output)._transform([](const auto  x){return x*LRATE;});
        
        b2 = b2 + b2._transform([&](const auto& x ){return x + d_predicted_output.sum();})._transform([](const auto  x){return x*LRATE;});

        w1 = w1 + d_hidden_layer.transpose().matmul(x)._transform([](const auto  x){return x*LRATE;});

        b1 = b1 + b1._transform([&](const auto& x ){return x + d_hidden_layer.sum();})._transform([](const auto  x){return x*LRATE;});

        if (i > 4995){
            predicted_output.print();   std::cout <<"====="<<std::endl;

        }
    }
    
    
}
