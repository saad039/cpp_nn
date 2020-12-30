#if !defined(UTIL_H)
#define UTIL_H

#include<tuple>

namespace util
{
    auto println = [](auto&&... args)  {
            (std::cout << ... << args);
            std::cout <<std::endl;
        };
    
    auto printshape = [](const str, const std::pair<std::size_t,std::size_t>& shape){
        println(str,"(",shape.first,",",shape.second,")");
    };
}

#endif // UTIL_H
