#ifndef TENSOR_H
#define TENSOR_H

#include "rng.h"
#include "util.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <execution>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <tuple>

#ifdef PARALLEL_UNSEQ_POL
#define PARALLEL_UNSEQ std::execution::par_unseq
#else
#define PARALLEL_UNSEQ
#endif

#ifdef PARALLEL_SEQ_POL
#define PARALLEL_SEQ std::execution::par
#else
#define PARALLEL_SEQ
#endif

#ifdef UNSEQ_POL
#define UNSEQ std::execution::unseq
#else
#define UNSEQ
#endif

/*
====================================================================
A wrapper arround std::array container which is populated with
random numbers sampled from the uniform distribution at compile time.
Other generator functions can be used to override this behaviour at
compile time but afte construction.

Random number generated ⊆  (MN,...,MX)
====================================================================
*/

template <typename T, std::size_t Rows, std::size_t Cols = 1, std::int32_t MN = 0,
          std::int32_t MX = 1>

class tensor {

    typedef T value_type;
    typedef decltype(Rows) size_type;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type* iterator;
    typedef const value_type* const_iterator;
    typedef std::array<value_type, Rows * Cols> container_type;

    typedef value_type (*generator)();

  private:
    constexpr static value_type MIN_VAL = static_cast<value_type>(MN);
    constexpr static value_type MAX_VAL = static_cast<value_type>(MX);

    constexpr static RNG<value_type, Rows * Cols> uniform_rn_gen{};
    container_type container{uniform_rn_gen(MIN_VAL, MAX_VAL)};

    constexpr static bool has_equal_shape(const size_type& Rows2, const size_type& Cols2) {
        return Rows == Rows2 and Cols == Cols2;
    }

    template <std::size_t Rows2, std::size_t Cols2, typename bin_op>
    [[nodiscard]] constexpr auto row_wise_arithmetic(const tensor<value_type, Rows2, Cols2>& other,
                                                     bin_op op) const noexcept {
        tensor<value_type, Rows, Cols> result;
        constexpr auto rowindices = util::array_iota(std::make_index_sequence<Rows>{});
        std::for_each(PARALLEL_UNSEQ, std::begin(rowindices), std::end(rowindices),
                      [&](const auto i) {
                          std::transform(UNSEQ begin() + i * Cols, begin() + ((i + 1) * Cols),
                                         std::begin(other), std::begin(result) + i * Cols, op);
                      });
        return result;
    }

  public:
    tensor(/* args */) = default;
    // template<T... Args>
    // [[nodiscard]] constexpr tensor(Args... vals) noexcept :container({vals...}){}

    [[nodiscard]] constexpr tensor(std::array<value_type, Rows * Cols>&& li) : container(li) {}
    ~tensor() = default;

    // begin and end iterators
    constexpr iterator begin() noexcept { return container.begin(); }
    constexpr const_iterator begin() const noexcept { return container.begin(); }

    constexpr iterator end() noexcept { return container.end(); }
    constexpr const_iterator end() const noexcept { return container.end(); }

    // operator[]
    constexpr reference operator[](size_type pos) { return container[pos]; }
    constexpr const_reference operator[](size_type pos) const { return container[pos]; }

    // sizes and capacity
    constexpr size_type capacity() const noexcept { return container.max_size(); }
    constexpr size_type size() const noexcept { return container.size(); }
    void shape() const noexcept { std::cout << '(' << Rows << ',' << Cols << ")\n"; }

    // Fill the tensor with constants.
    constexpr void constant_fill(const value_type& val) noexcept { std::fill(begin(), end(), val); }

    // Fill the tensor by invoking a function for each element
    constexpr void generator_fill(generator gen) noexcept {
        std::generate(PARALLEL_SEQ, begin(), end(), gen);
    }

    constexpr void range_fill(const_reference start) noexcept { std::iota(begin(), end(), start); }

    // Apply a transformation to each element in the tensor.
    template <typename unary_op>
    constexpr void transform(unary_op trn) noexcept {
        std::transform(PARALLEL_UNSEQ, begin(), end(), begin(), trn);
    }

    template <typename unary_op>
    constexpr auto _transform(unary_op trn) const noexcept {
        tensor<value_type, Rows, Cols> result;
        std::transform(PARALLEL_UNSEQ, begin(), end(), std::begin(result), trn);
        return result;
    }

    constexpr void tanh() noexcept { transform(std::tanh); }

    constexpr void ones() noexcept {
        generator_fill([]() { return static_cast<value_type>(1); });
    }

    template <std::size_t Rows2, std::size_t Cols2>
    [[nodiscard]] constexpr value_type
    dot(const tensor<value_type, Rows2, Cols2>& other) const noexcept {

        static_assert((Rows == Rows2) == 1 or (Cols == Cols2) == 1,
                      "tensors are not 1D"); // must be 1d arrays
        static_assert(has_equal_shape(Rows2, Cols2), "mismatch shapes");

        return std::transform_reduce(PARALLEL_UNSEQ, begin(), end(), other.begin(),
                                     static_cast<value_type>(0));
    }

    [[nodiscard]] constexpr value_type sum() const noexcept {
        return std::accumulate(begin(), end(), static_cast<value_type>(0));
    }

    template <std::size_t Rows2, std::size_t Cols2>
    [[nodiscard]] constexpr auto
    element_wise_mul(const tensor<value_type, Rows2, Cols2>& other) const noexcept {

        static_assert(Rows == Rows2 == 1 or Cols == Cols2 == 1,
                      "tensors are not 1D"); // must be 1d arrays
        static_assert(has_equal_shape(Rows2, Cols2), "mismatch shapes");

        tensor<value_type, Rows, Cols> result;
        std::transform(PARALLEL_UNSEQ, begin(), end(), other.begin(), result.begin(),
                       std::multiplies<>{});

        return result;
    }

    [[nodiscard]] constexpr auto element_wise_add_s(const value_type val) noexcept {
        std::transform(begin(), end(), begin(), [&](const auto x) { return val + x; });
        return *this;
    }

    template <std::size_t Rows2, std::size_t Cols2>
    [[nodiscard]] constexpr auto
    matmul(const tensor<value_type, Rows2, Cols2>& other) const noexcept {
        static_assert(Cols == Rows2, "mismatch matrix dimensions");
        tensor<value_type, Rows, Cols2> result;
        const auto tr_tensor = other.transpose();
        // After transpose, tr_tensor has 'Cols2' rows and 'Rows2' columns
        for (size_type i = 0; i < Rows; i++) {
            const auto start_1 = begin() + i * Cols;
            const auto end_1 = begin() + (i + 1) * Cols;
            for (size_type j = 0; j < Cols2; j++) {
                const auto start_2 = tr_tensor.begin() + (j * Rows2);
                result[j + Cols2 * i] = std::transform_reduce(PARALLEL_UNSEQ, start_1, end_1,
                                                              start_2, static_cast<value_type>(0));
            }
        }
        return result;
    }

    template <std::size_t Rows2, std::size_t Cols2>
    [[nodiscard]] constexpr auto
    operator+(const tensor<value_type, Rows2, Cols2>& other) const noexcept {

        if constexpr (Rows2 == 1 and Cols2 == Cols) { // other is a row vector; i.e {1,2,3,4,5,6}
            return row_wise_arithmetic(other, std::plus<>()); // Row wise addition  (other<1,N>)
        } else if constexpr (Cols2 == 1 and
                             Rows2 == Rows) { // other is a col vector; i.e ({1,2,3,4,5,6}).T
            return (this->transpose() + other.transpose())
                .transpose(); // Col wise addition    (other<N,1>)
        } else if constexpr (Rows == Rows2 and Cols == Cols2) {
            tensor<value_type, Rows, Cols> result;
            std::transform(PARALLEL_UNSEQ, begin(), end(), std::begin(other), std::begin(result),
                           std::plus<>{}); // Matrix Addition  (other<N,N>)
            return result;
        } else {
            throw std::logic_error("mismatch dimensions for addition");
        }
    }

    template <std::size_t Rows2, std::size_t Cols2>
    [[nodiscard]] constexpr auto
    operator-(const tensor<value_type, Rows2, Cols2>& other) const noexcept {

        if constexpr (Rows2 == 1 and Cols2 == Cols) { // other is a row vector; i.e {1,2,3,4,5,6}
            return row_wise_arithmetic(other, std::minus<>()); // Row wise addition  (other<1,N>)
        } else if constexpr (Cols2 == 1 and
                             Rows2 == Rows) { // other is a col vector; i.e ({1,2,3,4,5,6}).T
            return (this->transpose() - other.transpose())
                .transpose(); // Col wise addition    (other<N,1>)
        } else if constexpr (Rows == Rows2 and Cols == Cols2) {
            tensor<value_type, Rows, Cols> result;
            std::transform(PARALLEL_UNSEQ, begin(), end(), std::begin(other), std::begin(result),
                           std::minus<>{}); // Matrix Addition  (other<N,N>)
            return result;
        } else {
            throw std::logic_error("mismatch dimensions for addition");
        }
    }

    [[nodiscard]] constexpr tensor<value_type, Cols, Rows> transpose() const noexcept {
        tensor<value_type, Cols, Rows> result;

        for (size_type n = 0; n < Rows * Cols; n++) {
            const auto i = n / Rows;
            const auto j = n % Rows;
            result[n] = container[Cols * j + i];
        }
        return result;
    }

    void print() const noexcept {
        for (size_type i = 0; i < Rows; i++) {
            for (size_type j = 0; j < Cols; j++) { std::cout << container[j + Cols * i] << ','; }
            std::cout << '\n';
        }
    }

    constexpr auto& get_shape() const noexcept { return std::make_pair(Rows, Cols); }
};

using dtype = float;

template <std::size_t rows, std::size_t cols>
using tensor_t = tensor<dtype, rows, cols>;

#endif
