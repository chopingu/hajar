#pragma once

#include <limits>

namespace lmj::detail {
/**
 * @tparam n an unsigned int value
 * @return 0 with the smallest type which can represent n
 */
template<std::size_t n>
consteval auto needed_uint() {
    if constexpr (n <= std::numeric_limits<std::uint8_t>::max()) {
        return std::uint8_t{};
    } else if constexpr (n <= std::numeric_limits<std::uint16_t>::max()) {
        return std::uint16_t{};
    } else if constexpr (n <= std::numeric_limits<std::uint32_t>::max()) {
        return std::uint32_t{};
    } else {
        return std::uint64_t{};
    }
}

// tests

static_assert(sizeof(needed_uint<std::numeric_limits<std::uint8_t>::min()>()) == 1);
static_assert(sizeof(needed_uint<std::numeric_limits<std::uint8_t>::max()>()) == 1);
static_assert(sizeof(needed_uint<std::numeric_limits<std::uint16_t>::max()>()) == 2);
static_assert(sizeof(needed_uint<std::numeric_limits<std::uint32_t>::max()>()) == 4);
static_assert(sizeof(needed_uint<std::numeric_limits<std::uint64_t>::max()>()) == 8);
}