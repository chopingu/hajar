#include "include_all.hpp"

#include <cmath>

template<class ContainerType>
concept Container = requires(ContainerType a, const ContainerType b)
{
    requires std::regular<ContainerType>;
    requires std::swappable<ContainerType>;
    requires std::destructible<typename ContainerType::value_type>;
    requires std::same_as<typename ContainerType::reference, typename ContainerType::value_type &>;
    requires std::same_as<typename ContainerType::const_reference, const typename ContainerType::value_type &>;
    requires std::forward_iterator<typename ContainerType::iterator>;
    requires std::forward_iterator<typename ContainerType::const_iterator>;
    requires std::signed_integral<typename ContainerType::difference_type>;
    requires std::same_as<typename ContainerType::difference_type, typename std::iterator_traits<typename ContainerType::iterator>::difference_type>;
    requires std::same_as<typename ContainerType::difference_type, typename std::iterator_traits<typename ContainerType::const_iterator>::difference_type>;
    { a.begin() } -> std::same_as<typename ContainerType::iterator>;
    { a.end() } -> std::same_as<typename ContainerType::iterator>;
    { b.begin() } -> std::same_as<typename ContainerType::const_iterator>;
    { b.end() } -> std::same_as<typename ContainerType::const_iterator>;
    { a.cbegin() } -> std::same_as<typename ContainerType::const_iterator>;
    { a.cend() } -> std::same_as<typename ContainerType::const_iterator>;
    { a.size() } -> std::same_as<typename ContainerType::size_type>;
    { a.max_size() } -> std::same_as<typename ContainerType::size_type>;
    { a.empty() } -> std::same_as<bool>;
};

static_assert(Container<lmj::static_vector<int, 1>>);
static_assert(Container<lmj::static_hash_table<int, int, 1>>);
static_assert(Container<lmj::hash_table<int, int>>);

int main() {
    {
        // ensure basic functionality of lmj::hash_table
        lmj::hash_table<int, int> map;
        for (int i = 0; i < 1024; ++i)
            map[i] = i;
        for (int i = 0; i < 1024; i += 2)
            map.erase(i);
        int sum = 0;
        for (int i = 0; i < 1024; ++i)
            sum += map[i];
        assert(sum == 512 * 512); // sum of odd numbers up to and including 1023 = ((1023 - 1) / 2) ^ 2
        lmj::print("Test 1 passed!");
    }
    {
        // miscellaneous test of lmj::hash_table ("do random bullsh*t and see if the standard version ends up the same")
        constexpr int N = 1 << 10;
        std::unordered_map<int, int> map;
        lmj::hash_table<int, int> check;
        for (int i = 0; i < N; ++i) {
            int key = i / 2, value = i;
            if (i > N / 32)
                value = -i;
            map[key] = value;
            check[key] = value;
        }
        for (int i = 0; i < N; ++i) {
            const int key = lmj::rand<int>();
            map.erase(key);
            check.erase(key);
        }
        for (int i = 0; i < N; ++i) {
            const int key = lmj::rand<int>();
            const int val = lmj::rand<int>();
            map[key] = val;
            check[key] = val;
        }
        for (int i = 0; i < N; ++i) {
            const int key = lmj::rand<int>();
            map.erase(key);
            check.erase(key);
        }
        for (int i = 0; i < N; ++i) {
            const int key = lmj::rand<int>();
            const int val = lmj::rand<int>();
            map[key] = val;
            check[key] = val;
        }
        for (int i = 0; i < N; ++i) {
            const int key = lmj::rand<int>();
            for (int j = 0; j < 100; ++j) {
                const int val = lmj::rand<int>();
                map[key] = val;
                check[key] = val;
            }
        }
        for ([[maybe_unused]] auto &[key, val]: map)
            assert(check[key] == val);
        lmj::print("Test 2 passed!");
    }
    {
        // test removing elements from lmj::hash_table using std::unordered_map to ensure correctness
        lmj::hash_table<int, int> map;
        std::unordered_map<int, int> check;
        std::vector<int> vals;

        assert(map.size() == check.size());

        constexpr int n = 1 << 20;

        vals.reserve(n);
        for (int i = 0; i < n; ++i)
            vals.push_back(lmj::rand<int>());

        for (int i = 0; i < n; i += 2) {
            map[vals[i]] = 0xBADF00D;
            check[vals[i]] = 0xBADF00D;
        }

        assert(map.size() == check.size());

        for ([[maybe_unused]] auto &[key, val]: check)
            assert(map.at(key) == val);

        for (int i = 0; i < n; i += 2) {
            map.erase(vals[i]);
            check.erase(vals[i]);
        }

        for ([[maybe_unused]] auto &[key, val]: check)
            assert(map.at(key) == val);

        assert(map.size() == check.size());

        lmj::print("Test 3 passed!");
    }
    {
        // test copy constructor
        lmj::hash_table<int, int> m;
        for (int i = 0; i < 128; ++i)
            m[i] = i;
        for ([[maybe_unused]] auto &[key, value]: m) {
            assert(key == value);
        }
        lmj::hash_table<int, int> const m2 = m;
        for ([[maybe_unused]] auto &[key, value]: m2)
            assert(key == value);
        lmj::print("Test 4 passed!");
    }
    {
        // test lmj::hash_table with a custom hash function
        auto hash = [](int x) { return x * x; };
        lmj::hash_table<int, int, decltype(hash)> m;
        for (int i = 0; i < 1024; ++i)
            m[i] = i;
        for ([[maybe_unused]] auto &[key, value]: m) {
            assert(key == value);
        }
        lmj::print("Test 5 passed!");
    }
    {
        // test lmj::hash_table find
        lmj::hash_table<int, int> m;
        for (int i = 0; i < 1024; ++i)
            m[i] = i;
        for (int i = 0; i < 1024; ++i) {
            assert(m.find(i) != m.end());
            assert(m.find(i)->first == i);
        }
        assert(m.find(1024) == m.end());
        lmj::print("Test 6 passed!");
    }
    {
        // test lmj::hash_table with different types of keys and values and hash
        auto hash = [](int x) { return x * x; };
        auto str = [](int x) { return "to make it a long string " + std::to_string(x); };
        lmj::hash_table<int, std::string> m1;
        lmj::hash_table<int, std::string, decltype(hash)> m2{hash};
        for (int i = 0; i < 1024; ++i)
            m1[i] = m2[i] = str(i);
        for (int i = 0; i < 1024; ++i) {
            assert(m1.find(i) != m1.end());
            assert(m1.find(i)->first == i);
            assert(m1.find(i)->second == str(i));
        }
        assert(m1 == m2);
        assert(m1.find(1024) == m1.end());
        lmj::print("Test 7 passed!");
    }
    lmj::print("All tests passed!");
}
