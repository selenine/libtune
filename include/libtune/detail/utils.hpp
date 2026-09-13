#pragma once

#include <utility>

namespace tune::detail {
template <std::size_t N, typename F>
constexpr inline void unroll(F&& func) {
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (func.template operator()<Is>(), ...);
    }(std::make_index_sequence<N>{});
}

constexpr auto UNROLL = []() {
#if defined(__AVX512F__) || defined(__ARM_NEON__)
    return 4;
#elif defined(__AVX2__)
    return 2;
#else
    return 1;
#endif
}();
}  // namespace tune::detail
