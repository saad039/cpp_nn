#if !defined(RGN_H)
#define RGN_H // Constexpr PseudoRandom Number Generator

#include <array>
#include <cstdint>
#include <type_traits>

template <typename T>
concept floating_point = std::is_floating_point_v<T>;
template <floating_point dtype, std::size_t N>

class RNG {

    constexpr static std::uint32_t lce_a = 4096, lce_c = 150889, lce_m = 714025;

  private:
    constexpr auto time_from_string(const char* str, int offset) const noexcept {
        return static_cast<std::uint32_t>(str[offset] - '0') * 10 +
               static_cast<std::uint32_t>(str[offset + 1] - '0');
    }

    constexpr auto get_seed_constexpr() const noexcept {
        constexpr auto t = __TIME__;
        return time_from_string(t, 0) * 60 * 60 + time_from_string(t, 3) * 60 +
               time_from_string(t, 6);
    }

    constexpr std::uint32_t uniform_distribution(std::uint32_t& previous) const noexcept {
        previous = ((lce_a * previous + lce_c) % lce_m);
        return previous;
    }

    constexpr dtype uniform_distribution_n(std::uint32_t& previous) const noexcept {
        auto dst = uniform_distribution(previous);
        return static_cast<dtype>(dst) / lce_m;
    }

  public:
    constexpr auto operator()(const dtype& min, const dtype& max) const noexcept {
        std::array<dtype, N> dst{};
        auto previous = get_seed_constexpr();
        for (auto& el : dst)
            el = static_cast<dtype>(uniform_distribution_n(previous) * (max - min) + min);
        return dst;
    }
};

#endif // RGN_H
