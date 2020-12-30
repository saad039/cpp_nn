#ifndef TENSOR_H
#define TENSOR_H

#include <array>
#include<algorithm>
#include<execution>
#include<cmath>
#include<tuple>
#include<cassert>
#include<numeric>

template <typename T, std::size_t Rows, std::size_t Cols=1>
class tensor {


typedef T                                                       value_type;
typedef decltype(Rows)                                          size_type  ;
typedef value_type&                                             reference;
typedef const value_type&                                       const_reference;
typedef value_type*                                             pointer;
typedef const value_type*                                       const_pointer;
typedef value_type*                                             iterator;
typedef const value_type*                                       const_iterator;
typedef std::pair<size_type,size_type>                          shape_type;
typedef std::array<value_type,Rows*Cols>                        container_type;

typedef value_type (*generator)();
typedef value_type (*unary_op)(const value_type&);

private:
    container_type container;
    shape_type dims = std::make_pair(Rows,Cols);
    constexpr bool has_equal_shape(const tensor& other) const noexcept {return dims.first == other.dims.first and dims.second == other.dims.second;} 

public:
    tensor(/* args */) =default;
    ~tensor() = default;

//begin and end iterators
    constexpr iterator begin() noexcept {return container.begin();}
    constexpr const_iterator begin() const noexcept {return container.begin();}

    constexpr iterator end() noexcept {return container.end();}
    constexpr const_iterator end() const noexcept {return container.end();}
    
//operator[]
    constexpr reference operator[]( size_type pos ){return container[pos];}    
    constexpr const_reference operator[]( size_type pos ) const{return container[pos];}

//sizes and capacity
    constexpr size_type capacity() const noexcept {return container.max_size();}
    constexpr size_type size() const noexcept {return container.size();}
    void shape() const noexcept{std::cout<<'('<<dims.first<<','<<dims.second<<")\n";}

//Fill the tensor with constants. 
    constexpr void constant_fill(const value_type& val){std::fill(begin(),end(),val);}

//Fill the tensor by invoking a function for each element
    void generator_fill(generator gen) {std::generate(std::execution::par,begin(),end(),gen);}

    void range_fill(const_reference start)
    {
        std::iota(begin(),end(),start);
    }

//Apply a transformation to each element in the tensor.
    void transform(unary_op trn){ std::transform(std::execution::par,begin(),end(),begin(),trn);}

    void tanh(){transform(std::tanh);}

    void ones(){generator_fill([](){return static_cast<value_type>(1);});}

    template<std::size_t Rows2, std::size_t Cols2>
    [[nodiscard]] value_type dot(const tensor<value_type,Rows2,Cols2>& other) const noexcept
    {
        
        static_assert(Rows == Rows2 == 1 or Cols == Cols2 == 1,"tensors are not 1D"); //must be 1d arrays
        static_assert(has_equal_shape(other), "mismatch shapes");
        
        return std::transform_reduce(std::execution::par_unseq,begin(),end(),other.begin(),static_cast<value_type>(0));

    }

    
    template<std::size_t Rows2, std::size_t Cols2>
    [[nodiscard]] auto matmul(const tensor<value_type,Rows2,Cols2>& other) const
    {
        static_assert(Cols == Rows2, "mismatch matrix dimensions");
        tensor<value_type,Rows,Cols2> result;
        const auto tr_tensor = other.transpose();
        //After transpose, tr_tensor has 'Cols2' rows and 'Rows2' columns
        for(size_type i = 0; i < Rows; i++){
                const auto start_1 = begin() + i*Cols;
                const auto end_1 = begin() + (i+1)*Cols;
            for(size_type j =0; j < Cols2; j++){
                const auto start_2 = tr_tensor.begin() + (j*Rows2);                
                result[j + Cols2*i] = std::transform_reduce(std::execution::par_unseq, start_1,end_1,start_2,static_cast<value_type>(0));       
            }
        }
        return result;
    }


    [[nodiscard]] tensor<value_type,Cols,Rows> transpose() const
    {
        tensor<value_type,Cols,Rows> result;

        for(size_type n = 0; n < Rows*Cols ; n++)
        {
            const auto i = n/Rows;
            const auto j = n%Rows;
            result[n] = container[Cols*j + i]; 
        }
        return result;
    }

    void print() const noexcept
    {
        for(size_type i = 0; i < Rows; i++){
            for(size_type j = 0; j < Cols; j++)
            {
                std::cout << container[j + Cols*i]<<',';
            }
            std::cout <<'\n';
        }
    }

    const auto& get_shape() const{return this->dims;}
};

using dtype = float;

template<std::size_t rows, std::size_t cols>
using tensor_t = tensor<dtype,rows,cols>;

#endif