#pragma once

#include <utility>
#include <functional>
#include <cstdint>

#include "container_helpers.hpp"

namespace lmj {
template<class T>
struct hash {
    constexpr auto operator()(T x) const -> std::enable_if_t<std::is_integral_v<T>, T> {
        return x;
    }
};

template<class key_t, class value_t, std::size_t _table_capacity, class hash_t>
class static_hash_table_iterator;

template<class key_t, class value_t, std::size_t _table_capacity, class hash_t>
class static_hash_table_const_iterator;

template<class key_type, class value_type, std::size_t _capacity, class hash_type = lmj::hash<key_type>>
class static_hash_table {
    enum active_enum {
        INACTIVE = 0,
        ACTIVE = 1,
        TOMBSTONE = 2,
    };
public:
    static_assert(_capacity && "a _table_capacity of zero is not allowed");
    using pair_type = std::pair<key_type, value_type>;
    using size_type = decltype(detail::needed_uint<_capacity>());
    using bool_type = std::uint8_t;
    using iterator = static_hash_table_iterator<key_type, value_type, _capacity, hash_type>;
    using const_iterator = static_hash_table_const_iterator<key_type, value_type, _capacity, hash_type>;
    pair_type m_table[_capacity]{};
    bool_type m_is_set[_capacity]{};
    size_type m_elem_count{};
    hash_type m_hasher{};

    constexpr static_hash_table() = default;

    constexpr static_hash_table(std::initializer_list<pair_type> l) {
        for (auto &p: l) emplace(std::move(p));
    }

    constexpr static_hash_table(static_hash_table const &other) { *this = other; }

    constexpr explicit static_hash_table(hash_type hasher) : m_hasher{hasher} {}

    ~static_hash_table() = default;

    constexpr static_hash_table &operator=(static_hash_table const &other) {
        if (this != &other)
            _copy(other);
        return *this;
    }

    constexpr bool operator==(static_hash_table const &other) const {
        if (other.size() != this->size())
            return false;
        for (size_type i = 0; i < _capacity; ++i) {
            if (m_is_set[i] == ACTIVE &&
                other.contains(m_table[i].first) &&
                other.at(m_table[i].first) != m_table[i].second) {
                return false;
            }
        }
        return true;
    }

    /**
     * @return reference to value associated with key or default constructs value if it doesn't exist
     */
    constexpr value_type &operator[](key_type const &key) {
        return get(key);
    }

    /**
     * @return value at key or fails
     */
    [[nodiscard]] constexpr value_type const &at(key_type const &key) const {
        const size_type idx = _get_index_read(key);
        assert(m_is_set[idx] == ACTIVE && m_table[idx].first == key && "key not found");
        return m_table[idx].second;
    }

    /**
     * @return value at key or fails
     */
    [[nodiscard]] constexpr value_type &at(key_type const &key) {
        const size_type idx = _get_index_read(key);
        assert(m_is_set[idx] == ACTIVE && m_table[idx].first == key && "key not found");
        return m_table[idx].second;
    }

    /**
     * @brief gets value at key or creates new value at key with default value
     * @return reference to value associated with key
     */
    constexpr value_type &get(key_type const &key) {
        if (!m_elem_count)
            return emplace(key, value_type{});
        const size_type idx = _get_index_read(key);
        return (m_is_set[idx] == ACTIVE && m_table[idx].first == key) ?
               m_table[idx].second : emplace(key, value_type{});
    }

    /**
     * @return whether key is in table
     */
    constexpr bool contains(key_type const &key) const {
        const size_type idx = _get_index_read(key);
        return m_is_set[idx] == ACTIVE && m_table[idx].first == key;
    }

    /**
     * @param key key which is removed from table
     */
    constexpr void erase(key_type const &key) {
        remove(key);
    }

    /**
     * @param key key which is removed from table
     */
    constexpr void remove(key_type const &key) {
        const size_type idx = _get_index_read(key);
        if (m_is_set[idx] == ACTIVE && m_table[idx].first == key) {
            --m_elem_count;
            m_table[idx].first = key_type{};
            m_table[idx].second = value_type{};
            m_is_set[idx] = TOMBSTONE;
        }
    }

    [[nodiscard]] constexpr auto begin() {
        return iterator(this, _get_start_index());
    }

    [[nodiscard]] constexpr auto end() {
        return iterator(this, _get_end_index());
    }

    [[nodiscard]] constexpr auto begin() const {
        return const_iterator(this, _get_start_index());
    }

    [[nodiscard]] constexpr auto end() const {
        return const_iterator(this, _get_end_index());
    }

    [[nodiscard]] constexpr auto cbegin() const {
        return const_iterator(this, _get_start_index());
    }

    [[nodiscard]] constexpr auto cend() const {
        return const_iterator(this, _get_end_index());
    }

    constexpr value_type &insert(pair_type const &pair) {
        return emplace(pair);
    }

    /**
     * @param pack arguments for constructing element
     * @return  reference to newly constructed value
     */
    template<class... Args>
    constexpr value_type &emplace(Args &&... pack) {
        static_assert(sizeof...(pack));
        assert(m_elem_count < _capacity);
        auto p = pair_type{std::forward<Args>(pack)...};
        const size_type hash = _get_hash(p.first);
        size_type idx = _get_index_read(p.first, hash);
        if (m_is_set[idx] == ACTIVE && m_table[idx].first == p.first)
            return m_table[idx].second;
        idx = _get_writable_index(p.first, hash);
        ++m_elem_count;
        m_is_set[idx] = ACTIVE;
        m_table[idx] = std::move(p);
        return m_table[idx].second;
    }

    /**
     * @return number of elements
     */
    [[nodiscard]] constexpr size_type size() const {
        return m_elem_count;
    }

    /**
     * @return _table_capacity of table
     */
    [[nodiscard]] constexpr size_type capacity() const {
        return _capacity;
    }

    /**
     * @brief remove all elements
     */
    constexpr void clear() {
        for (size_type i = 0; i < _capacity; ++i) {
            if (m_is_set[i] == ACTIVE) {
                m_table[i].~pair_type();
            }
            m_is_set[i] = INACTIVE;
        }
        m_elem_count = 0;
    }

    [[nodiscard]] constexpr const_iterator find(key_type const &key) const {
        if (!m_elem_count)
            return end();
        const size_type idx = _get_index_read(key);
        if (m_is_set[idx] == ACTIVE && m_table[idx].first == key)
            return const_iterator(this, idx);
        return end();
    }

    [[nodiscard]] constexpr bool empty() const {
        return m_elem_count == 0;
    }

private:
    [[nodiscard]] constexpr size_type _get_start_index() const {
        if (!m_elem_count)
            return 0;
        for (size_type i = 0; i < _capacity; ++i)
            if (m_is_set[i] == ACTIVE)
                return i;
        return 0; // should be unreachable;
    }

    [[nodiscard]] constexpr size_type _get_end_index() const {
        return _capacity;
    }

    constexpr void _copy(static_hash_table const &other) {
        for (size_type i = 0; i < other.capacity(); ++i) {
            if (other.m_is_set[i] == ACTIVE) {
                m_table[i].first = other.m_table[i].first;
                m_table[i].second = other.m_table[i].second;
            }
            m_is_set[i] = other.m_is_set[i];
        }
        m_elem_count = other.m_elem_count;
        if constexpr (std::is_copy_assignable_v<hash_type>)
            m_hasher = other.m_hasher;
    }

    [[nodiscard]] constexpr size_type _clamp_size(size_type idx) const {
        if constexpr (_capacity & (_capacity - 1))
            return idx % _capacity;
        else
            return idx & (_capacity - 1);
    }

    [[nodiscard]] constexpr size_type _get_hash(key_type const &key) const {
        const size_type hash = m_hasher(key);
        return _clamp_size(hash ^ (~hash >> 16) ^ (hash << 24));

    }

    [[nodiscard]] constexpr size_type _new_idx(size_type const idx) const {
        return _clamp_size(idx + 1);
    }

    [[nodiscard]] constexpr size_type _get_index_read(key_type const &key) const {
        return _get_index_read_impl(key, _get_hash(key));
    }

    [[nodiscard]] constexpr size_type _get_index_read(key_type const &key, size_type const idx) const {
        return _get_index_read_impl(key, idx);
    }

    [[nodiscard]] constexpr size_type _get_index_read_impl(key_type const &key, size_type idx) const {
        std::size_t _iterations = 0;
        while ((m_is_set[idx] == TOMBSTONE || (m_is_set[idx] == ACTIVE && m_table[idx].first != key)) &&
               _iterations++ < _capacity) {
            idx = _new_idx(idx);
        }
        return idx;
    }

    [[nodiscard]] constexpr size_type _get_writable_index(key_type const &key) const {
        return _get_writable_index_impl(key, _get_hash(key));
    }

    [[nodiscard]] constexpr size_type _get_writable_index(key_type const &key, size_type idx) const {
        return _get_writable_index_impl(key, idx);
    }

    [[nodiscard]] constexpr size_type _get_writable_index_impl(key_type const &key, size_type idx) const {
        [[maybe_unused]] std::size_t iterations = 0;
        while (m_is_set[idx] == ACTIVE && m_table[idx].first != key) {
            assert(iterations++ < _capacity && "empty slot not found");
            idx = _new_idx(idx);
        }
        return idx;
    }
};

template<class key_t, class value_t, std::size_t _table_capacity, class hash_t>
class static_hash_table_iterator {
    enum active_enum {
        INACTIVE = 0,
        ACTIVE = 1,
        TOMBSTONE = 2,
    };
public:
    using pair_type = std::pair<key_t, value_t>;
    using size_type = std::size_t;

    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = long long;
    using value_type = pair_type;
    using pointer = pair_type *;
    using reference = pair_type &;

    static_hash_table<key_t, value_t, _table_capacity, hash_t> *const m_table_ptr;
    size_type m_index;

    constexpr static_hash_table_iterator(static_hash_table<key_t, value_t, _table_capacity, hash_t> *ptr,
                                         size_type idx) : m_table_ptr{ptr}, m_index{idx} {}

    constexpr auto &operator++() {
        ++m_index;
        while (m_index < m_table_ptr->capacity() && m_table_ptr->m_is_set[m_index] != ACTIVE) ++m_index;
        return *this;
    }

    constexpr auto &operator--() {
        --m_index;
        while (m_index > 0 && m_table_ptr->m_is_set[m_index] != ACTIVE) --m_index;
        return *this;
    }

    constexpr reference operator*() const {
        return m_table_ptr->m_table[m_index];
    }

    constexpr auto operator->() const {
        return &m_table_ptr->m_table[m_index];
    }

    template<class T>
    constexpr bool operator!=(T const &other) const {
        return m_index != other.m_index || m_table_ptr != other.m_table_ptr;
    }

    template<class T>
    constexpr bool operator==(T other) const {
        return m_index == other.m_index || m_table_ptr == other.m_table_ptr;
    }
};

template<class key_t, class value_t, std::size_t _table_capacity, class hash_t>
class static_hash_table_const_iterator {
    enum active_enum {
        INACTIVE = 0,
        ACTIVE = 1,
        TOMBSTONE = 2,
    };
public:
    using pair_type = std::pair<key_t, value_t>;
    using size_type = std::size_t;

    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = long long;
    using value_type = pair_type const;
    using pointer = pair_type const *;
    using reference = pair_type const &;

    static_hash_table<key_t, value_t, _table_capacity, hash_t> const *const m_table_ptr;
    size_type m_index;

    constexpr static_hash_table_const_iterator(
            static_hash_table<key_t, value_t, _table_capacity, hash_t> const *ptr,
            size_type idx) : m_table_ptr{ptr}, m_index{idx} {}

    constexpr auto &operator++() {
        ++m_index;
        while (m_index < m_table_ptr->capacity() && m_table_ptr->m_is_set[m_index] != ACTIVE)
            ++m_index;
        return *this;
    }

    constexpr auto &operator--() {
        --m_index;
        while (m_index > 0 && m_table_ptr->m_is_set[m_index] != ACTIVE)
            --m_index;
        return *this;
    }

    constexpr reference operator*() const {
        return m_table_ptr->m_table[m_index];
    }

    constexpr auto operator->() const {
        return &m_table_ptr->m_table[m_index];
    }

    template<class T>
    constexpr bool operator!=(T const &other) const {
        return m_index != other.m_index || m_table_ptr != other.m_table_ptr;
    }

    template<class T>
    constexpr bool operator==(T other) const {
        return m_index == other.m_index || m_table_ptr == other.m_table_ptr;
    }
};

// tests

static_assert([] {
    static_hash_table<int, int, 128> map;
    for (int i = 0; i < 50; ++i)
        map[i] = i;
    auto res = 0;
    for (int i = 0; i < 50; ++i)
        res += map.at(i);
    return res;
}() == 50 * 49 / 2);

static_assert([] {
    lmj::static_hash_table<short, int, 128> map;
    for (int i = 0; i < 64; ++i) {
        map[static_cast<short>(i)] = i * i * i * i;
    }
    const auto map2 = map;
    return map == map2;
}());

static_assert([] {
    auto table_1 = [] {
        lmj::static_hash_table<int, int, 128> t;
        for (int i = 0; i < 100; ++i)
            t[i] = i;
        return t;
    }();
    auto table_2 = [] {
        lmj::static_hash_table<int, int, 128> t;
        int random_nums[100]{};
        std::size_t state = 8662772801;
        for (auto &random_num: random_nums) {
            state = state * 7967335919 + 1078795391;
            random_num = static_cast<int>(state & 63);
            t[random_num] = 0xBADF00D;
        }
        for (auto random_num: random_nums)
            t.erase(random_num);
        for (int i = 0; i < 100; ++i) {
            t[i] = i;
        }
        return t;
    }();
    return table_1 == table_2;
}());

static_assert([] {
    constexpr auto m = [] {
        lmj::static_hash_table<int, int, 2> t;
        t[2] = 0;
        t[4] = 0;
        t.erase(2);
        t[1] = 1;
        return t;
    }();
    return m.at(1) == 1;
}());
}
