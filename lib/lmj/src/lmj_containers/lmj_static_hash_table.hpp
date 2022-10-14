#pragma once

#include <utility>
#include <functional>
#include <cstdint>

#include "lmj_container_helpers.hpp"

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
    pair_type _table[_capacity]{};
    bool_type _is_set[_capacity]{};
    size_type _elem_count{};
    hash_type _hasher{};

    constexpr static_hash_table() = default;

    constexpr static_hash_table(std::initializer_list<pair_type> l) {
        for (auto &p: l) emplace(std::move(p));
    }

    constexpr static_hash_table(static_hash_table const &other) { *this = other; }

    constexpr explicit static_hash_table(hash_type _hasher) : _hasher{_hasher} {}

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
            if (_is_set[i] == ACTIVE &&
                other.contains(_table[i].first) &&
                other.at(_table[i].first) != _table[i].second) {
                return false;
            }
        }
        return true;
    }

    /**
     * @return reference to value associated with _key or default constructs value if it doesn't exist
     */
    constexpr value_type &operator[](key_type const &_key) {
        return get(_key);
    }

    /**
     * @return value at _key or fails
     */
    constexpr value_type const &at(key_type const &_key) const {
        size_type _idx = _get_index_read(_key);
        assert(_is_set[_idx] == ACTIVE && _table[_idx].first == _key && "key not found");
        return _table[_idx].second;
    }

    /**
     * @brief gets value at _key or creates new value at _key with default value
     * @return reference to value associated with _key
     */
    constexpr value_type &get(key_type const &_key) {
        if (!_elem_count)
            return emplace(_key, value_type{});
        size_type _idx = _get_index_read(_key);
        return (_is_set[_idx] == ACTIVE && _table[_idx].first == _key) ?
               _table[_idx].second : emplace(_key, value_type{});
    }

    /**
     * @return whether _key is in table
     */
    constexpr bool contains(key_type const &_key) const {
        size_type _idx = _get_index_read(_key);
        return _is_set[_idx] == ACTIVE && _table[_idx].first == _key;
    }

    /**
     * @param _key key which is removed from table
     */
    constexpr void erase(key_type const &_key) {
        remove(_key);
    }

    /**
     * @param _key key which is removed from table
     */
    constexpr void remove(key_type const &_key) {
        size_type _idx = _get_index_read(_key);
        if (_is_set[_idx] == ACTIVE && _table[_idx].first == _key) {
            --_elem_count;
            _table[_idx].first = key_type{};
            _table[_idx].second = value_type{};
            _is_set[_idx] = TOMBSTONE;
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

    constexpr value_type &insert(pair_type const &_pair) {
        return emplace(_pair);
    }

    /**
     * @param _pack arguments for constructing element
     * @return  reference to newly constructed value
     */
    template<class... T>
    constexpr value_type &emplace(T &&... _pack) {
        static_assert(sizeof...(_pack));
        assert(_elem_count < _capacity);
        auto _p = pair_type{std::forward<T>(_pack)...};
        const size_type _hash = _get_hash(_p.first);
        size_type _idx = _get_index_read(_p.first, _hash);
        if (_is_set[_idx] == ACTIVE && _table[_idx].first == _p.first)
            return _table[_idx].second;
        _idx = _get_writable_index(_p.first, _hash);
        ++_elem_count;
        _is_set[_idx] = ACTIVE;
        _table[_idx] = std::move(_p);
        return _table[_idx].second;
    }

    /**
     * @return number of elements
     */
    [[nodiscard]] constexpr size_type size() const {
        return _elem_count;
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
            if (_is_set[i] == ACTIVE) {
                _table[i].~pair_type();
            }
            _is_set[i] = INACTIVE;
        }
        _elem_count = 0;
    }

    [[nodiscard]] constexpr const_iterator find(key_type const &_key) const {
        if (!_elem_count)
            return end();
        size_type _idx = _get_index_read(_key);
        if (_is_set[_idx] == ACTIVE && _table[_idx].first == _key)
            return const_iterator(this, _idx);
        return end();
    }

private:
    [[nodiscard]] constexpr size_type _get_start_index() const {
        if (!_elem_count)
            return 0;
        for (size_type i = 0; i < _capacity; ++i)
            if (_is_set[i] == ACTIVE)
                return i;
        return 0; // should be unreachable;
    }

    [[nodiscard]] constexpr size_type _get_end_index() const {
        return _capacity;
    }

    constexpr void _copy(static_hash_table const &other) {
        for (size_type i = 0; i < other.capacity(); ++i) {
            if (other._is_set[i] == ACTIVE) {
                _table[i].first = other._table[i].first;
                _table[i].second = other._table[i].second;
            }
            _is_set[i] = other._is_set[i];
        }
        _elem_count = other._elem_count;
        if constexpr (std::is_copy_assignable_v<hash_type>)
            _hasher = other._hasher;
    }

    [[nodiscard]] constexpr size_type _clamp_size(size_type _idx) const {
        if constexpr (_capacity & (_capacity - 1))
            return _idx % _capacity;
        else
            return _idx & (_capacity - 1);
    }

    [[nodiscard]] constexpr size_type _get_hash(key_type const &_key) const {
        const size_type _hash = _hasher(_key);
        return _clamp_size(_hash ^ (~_hash >> 16) ^ (_hash << 24));

    }

    [[nodiscard]] constexpr size_type _new_idx(size_type const _idx) const {
        return _clamp_size(_idx + 1);
    }

    [[nodiscard]] constexpr size_type _get_index_read(key_type const &_key) const {
        return _get_index_read_impl(_key, _get_hash(_key));
    }

    [[nodiscard]] constexpr size_type _get_index_read(key_type const &_key, size_type _idx) const {
        return _get_index_read_impl(_key, _idx);
    }

    [[nodiscard]] constexpr size_type _get_index_read_impl(key_type const &_key, size_type _idx) const {
        std::size_t _iterations = 0;
        while ((_is_set[_idx] == TOMBSTONE || (_is_set[_idx] == ACTIVE && _table[_idx].first != _key)) &&
               _iterations++ < _capacity) {
            _idx = _new_idx(_idx);
        }
        return _idx;
    }

    [[nodiscard]] constexpr size_type _get_writable_index(key_type const &_key) const {
        return _get_writable_index_impl(_key, _get_hash(_key));
    }

    [[nodiscard]] constexpr size_type _get_writable_index(key_type const &_key, size_type _idx) const {
        return _get_writable_index_impl(_key, _idx);
    }

    [[nodiscard]] constexpr size_type _get_writable_index_impl(key_type const &_key, size_type _idx) const {
        std::size_t _iterations = 0;
        while (_is_set[_idx] == ACTIVE && _table[_idx].first != _key) {
            assert(_iterations++ < _capacity && "empty slot not found");
            _idx = _new_idx(_idx);
        }
        return _idx;
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

    static_hash_table<key_t, value_t, _table_capacity, hash_t> *const _table_ptr;
    size_type _index;

    constexpr static_hash_table_iterator(static_hash_table<key_t, value_t, _table_capacity, hash_t> *_ptr,
                                         size_type _idx) : _table_ptr{_ptr}, _index{_idx} {}

    constexpr auto &operator++() {
        ++_index;
        while (_index < _table_ptr->capacity() && _table_ptr->_is_set[_index] != ACTIVE) ++_index;
        return *this;
    }

    constexpr auto &operator--() {
        --_index;
        while (_index > 0 && _table_ptr->_is_set[_index] != ACTIVE) --_index;
        return *this;
    }

    constexpr reference operator*() const {
        return _table_ptr->_table[_index];
    }

    constexpr auto operator->() const {
        return &_table_ptr->_table[_index];
    }

    constexpr bool operator!=(static_hash_table_iterator const &other) const {
        return _index != other._index || _table_ptr != other._table_ptr;
    }

    template<class T>
    constexpr bool operator==(T const &other) const {
        return !(*this != other);
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

    static_hash_table<key_t, value_t, _table_capacity, hash_t> const *const _table_ptr;
    size_type _index;

    constexpr static_hash_table_const_iterator(
            static_hash_table<key_t, value_t, _table_capacity, hash_t> const *_ptr,
            size_type _idx) : _table_ptr{_ptr}, _index{_idx} {}

    constexpr auto &operator++() {
        ++_index;
        while (_index < _table_ptr->capacity() && _table_ptr->_is_set[_index] != ACTIVE) ++_index;
        return *this;
    }

    constexpr auto &operator--() {
        --_index;
        while (_index > 0 && _table_ptr->_is_set[_index] != ACTIVE) --_index;
        return *this;
    }

    constexpr reference operator*() const {
        return _table_ptr->_table[_index];
    }

    constexpr auto operator->() const {
        return &_table_ptr->_table[_index];
    }

    constexpr bool operator!=(static_hash_table_const_iterator const &other) const {
        return _index != other._index || _table_ptr != other._table_ptr;
    }

    template<class T>
    constexpr bool operator==(T const &other) const {
        return !(*this != other);
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
