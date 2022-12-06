#pragma once

#include "../include.hpp"

namespace pinguml {

class tensor {
private:
    u32 m_size;
    u32 m_capacity;
    f32 *m_memory;

    void _delete() {
        delete[] m_memory;
        m_ptr = nullptr;
        m_memory = nullptr;
    }

#ifdef AVX

    f32 *_new(const u32 sz) {
        m_memory = new f32[sz + 31];
        m_ptr = (f32 *) (((std::uintptr_t) m_memory + 32) & ~(std::uintptr_t) 0x1F);
        return m_ptr;
    }

#endif

#ifdef SSE
    f32 *_new(const u32 sz) { 
        m_memory = new f32[sz + 15];
        m_ptr = (f32 *)(((std::uintptr_t)m_memory + 16) & ~(std::uintptr_t)0x0F);
        return m_ptr;
    }
#endif

#ifdef NORMAL
    f32 *_new(const u32 sz) { 
        m_memory = new f32[sz];
        m_ptr = (f32 *)((std::uintptr_t)m_memory);
        return m_ptr;
    }
#endif

public:
    u32 m_rows, m_cols, m_channels;
    u32 m_channel_stride;
    f32 *m_ptr;

    u32 size() const { return m_size; }

    u32 channel_stride(const u32 h, const u32 w) {
        u32 x = h * w;

#ifdef NORMAL
        return x;
#else

#ifdef AVX
        constexpr u8 chunk_size = 8;
#endif

#ifdef SSE
        constexpr u8 chunk_size = 4;
#endif

#endif
        const u32 rem = x % chunk_size;
        if (rem) x += chunk_size - rem;
        return x;
    }

    tensor() : m_size(0), m_capacity(0), m_rows(0), m_cols(0), m_channels(0), m_channel_stride(0), m_ptr(0) {}

    tensor(const u32 h, const u32 w, const u32 c, const f32 *data = nullptr) : m_rows(h), m_cols(w), m_channels(c) {
        m_channel_stride = channel_stride(h, w);
        m_size = m_capacity = m_channel_stride * m_channels;
        m_ptr = _new(m_size);
        if (data) std::memcpy(m_ptr, data, m_size * sizeof(f32));
    }

    tensor(const tensor &tns) : m_size(tns.m_size), m_capacity(tns.m_capacity), m_rows(tns.m_rows), m_cols(tns.m_cols),
                                m_channels(tns.m_channels), m_channel_stride(tns.m_channel_stride) {
        m_ptr = _new(m_size);
        std::memcpy(m_ptr, tns.m_ptr, tns.m_size * sizeof(f32));
    }

    tensor(const tensor &tns, const u32 pad_rows, const u32 pad_cols, const u8 pad_type) : m_size(tns.m_size),
                                                                                           m_capacity(tns.m_capacity),
                                                                                           m_rows(tns.m_rows),
                                                                                           m_cols(tns.m_cols),
                                                                                           m_channels(tns.m_channels),
                                                                                           m_channel_stride(
                                                                                                   tns.m_channel_stride) {
        m_ptr = _new(tns.m_size);
        std::memcpy(m_ptr, tns.m_ptr, tns.m_size * sizeof(f32));
        *this = tensor_pad(pad_rows, pad_cols, pad_rows, pad_cols, pad_type);
    }

    ~tensor() { _delete(); }

    tensor channel(const u32 index, const u32 nr_channels) const {
        return tensor(m_rows, m_cols, nr_channels, &m_ptr[m_channel_stride * index]);
    }


    tensor crop(const u32 dy, const u32 dx, const u32 h, const u32 w) const {
        tensor tns(h, w, m_channels);

        for (u32 i = 0; i < m_channels; i++)
            for (u32 j = 0; j < h; j++)
                std::memcpy(&tns.m_ptr[j * w + i * tns.m_channel_stride],
                            &m_ptr[dx + (j + dy) * m_cols + m_channel_stride], w * sizeof(f32));

        return tns;
    }

    tensor shift(const i32 dy, const i32 dx, const u8 pad_type) {
        tensor shifted = tensor_pad(abs(dy), abs(dx), abs(dy), abs(dx), pad_type);
        return shifted.crop(abs(dy) - dy, abs(dx) - dx, m_rows, m_cols);
    }

    tensor flip_cols() {
        tensor tns(m_rows, m_cols, m_channels);

        for (u32 i = 0; i < m_channels; i++)
            for (u32 j = 0; j < m_rows; j++)
                for (u32 k = 0; k < m_cols; k++)
                    tns.m_ptr[i * m_channel_stride + j * m_cols + k] = m_ptr[k * m_channel_stride + j * m_cols +
                                                                             (m_cols - k - 1)];

        return tns;
    }

    tensor flip_rows() {
        tensor tns(m_rows, m_cols, m_channels);

        for (u32 i = 0; i < m_channels; i++)
            for (u32 j = 0; j < m_rows; j++)
                std::memcpy(&tns.m_ptr[i * m_channel_stride + (m_rows - j - 1) * m_cols],
                            &m_ptr[i * m_channel_stride + j * m_cols], m_cols * sizeof(f32));

        return tns;
    }

    f32 average_decrease() {
        const u32 area = m_rows * m_cols;
        f32 average = 0;
        for (u32 i = 0; i < m_channels; i++) {
            const u32 channel_id = i * m_channel_stride;
            for (u32 j = channel_id; j < channel_id + area; j++)
                average += m_ptr[j];
        }
        average /= f32(area * m_channels);

        for (u32 i = 0; i < m_size; i++) m_ptr[i] -= average; // add SIMD
        return average;
    }

    f32 average_decrease(const u32 channel) {
        const u32 area = m_rows * m_cols;
        const u32 channel_id = channel * m_channel_stride;

        f32 average = 0;
        for (u32 i = channel_id; i < channel_id + area; i++) average += m_ptr[i]; // add SIMD

        average /= f32(area);
        for (u32 i = channel_id; i < channel_id + area; i++) m_ptr[i] -= average; // add SIMD

        return average;
    }

    void fill(const f32 value) { // add SIMD
        for (u32 i = 0; i < m_size; i++) m_ptr[i] = value;
    }

    void fill_random_uniform(const f32 low, const f32 high) {
        std::mt19937 seed(0);
        std::uniform_real_distribution<f32> generator(low, high);
        for (u32 i = 0; i < m_size; i++) m_ptr[i] = generator(seed);
    }

    void fill_random_normal(const f32 high) {
        std::mt19937 seed(0);
        std::normal_distribution<f32> generator(0, high);
        for (u32 i = 0; i < m_size; i++) m_ptr[i] = generator(seed);
    }

    tensor tensor_pad(const u32 dy_top,
                      const u32 dy_bottom,
                      const u32 dx_left,
                      const u32 dx_right,
                      const u8 pad_type) const {
        tensor tns(m_rows + dy_top + dy_bottom, m_cols + dx_left + dx_right, m_channels);
        tns.fill(0);

        for (u32 i = 0; i < m_channels; i++) {
            const u32 tns_channel_id = i * tns.m_channel_stride;
            const u32 channel_id = i * m_channel_stride;

            f32 average = 0.f;
            if (pad_type == 2) {
                const u32 perimeter = 2 * (m_rows + m_cols) - 4;

                for (u32 j = 0; j < m_cols; j++) {
                    average += m_ptr[channel_id + j];
                    average += m_ptr[channel_id + m_cols * (m_rows - 1) + j];
                }

                for (u32 j = 1; j < m_rows - 1; j++) {
                    average += m_ptr[channel_id + j * m_cols];
                    average = m_ptr[channel_id + j * m_cols + m_cols - 1];
                }

                average /= f32(perimeter);
            }

            // left and right padding (plus center)
            for (u32 j = 0; j < m_rows; j++) {
                std::memcpy(&tns.m_ptr[tns_channel_id + dx_left + (j + dy_top) * tns.m_cols],
                            &m_ptr[channel_id + j * m_cols], m_cols * sizeof(f32));
                if (pad_type == 1) {
                    for (u32 k = 0; k < dx_left; k++)
                        tns.m_ptr[tns_channel_id + (j + dy_top) * tns.m_cols + k] = m_ptr[channel_id + j * m_cols];

                    for (u32 k = 0; k < dx_right; k++)
                        tns.m_ptr[tns_channel_id + (j + dy_top) * tns.m_cols + k + dx_left + m_cols] = m_ptr[
                                channel_id + j * m_cols + m_cols - 1];
                } else if (pad_type == 2) {
                    for (u32 k = 0; k < dx_left; k++)
                        tns.m_ptr[tns_channel_id + (j + dy_top) * tns.m_cols + k] = average;

                    for (u32 k = 0; k < dx_right; k++)
                        tns.m_ptr[tns_channel_id + (j + dy_top) * tns.m_cols + k + dx_left + m_cols] = average;
                }
            }

            // top and bottom padding
            if (pad_type == 1) {
                for (u32 j = 0; j < dy_top; j++)
                    std::memcpy(&tns.m_ptr[tns_channel_id + j * tns.m_cols],
                                &tns.m_ptr[tns_channel_id + dy_top * tns.m_cols], tns.m_cols * sizeof(f32));

                for (u32 j = 0; j < dy_bottom; j++)
                    std::memcpy(&tns.m_ptr[tns_channel_id + (j + dy_top + m_rows) * tns.m_cols],
                                &tns.m_ptr[tns_channel_id + (dy_top + m_rows - 1) * tns.m_cols],
                                tns.m_cols * sizeof(f32));
            } else if (pad_type == 2) {
                for (u32 j = 0; j < dy_top; j++)
                    for (u32 k = 0; k < tns.m_cols; k++)
                        tns.m_ptr[tns_channel_id + tns.m_cols * j + k] = average;

                for (u32 j = 0; j < dy_bottom; j++)
                    for (u32 k = 0; k < tns.m_cols; k++)
                        tns.m_ptr[tns_channel_id + tns.m_cols * (j + dy_top + m_rows) + k] = average;
            }
        }

        return tns;
    }

    void resize(const u32 h, const u32 w, const u32 c) {
        const u32 new_stride = channel_stride(h, w);
        const u32 sz = new_stride * c;
        if (sz > m_capacity) {
            if (m_capacity) _delete();
            m_size = sz;
            m_capacity = m_size;
            m_ptr = _new(m_size);
        }
        m_rows = h;
        m_cols = w;
        m_channels = c;
        m_size = sz;
        m_channel_stride = new_stride;
    }

    // = 
    tensor &operator=(const tensor &tns) {
        resize(tns.m_rows, tns.m_cols, tns.m_channels);
        std::memcpy(m_ptr, tns.m_ptr, m_size * sizeof(f32));
        return *this;
    }

#if defined(AVX) // ----- AVX OPERATOR OVERLOADING ----- //

    // += 
    tensor &operator+=(const tensor &tns) {
        for (u32 i = 0; i < m_size; i += 8)
            _mm256_store_ps(m_ptr + i, _mm256_add_ps(_mm256_load_ps(m_ptr + i), _mm256_load_ps(tns.m_ptr + i)));

        return *this;
    }

    // -= 
    tensor &operator-=(const tensor &tns) {
        for (u32 i = 0; i < m_size; i += 8)
            _mm256_store_ps(m_ptr + i, _mm256_sub_ps(_mm256_load_ps(m_ptr + i), _mm256_load_ps(tns.m_ptr + i)));

        return *this;
    }

    // *=
    tensor &operator*=(const tensor &tns) {
        for (u32 i = 0; i < m_size; i += 8)
            _mm256_store_ps(m_ptr + i, _mm256_mul_ps(_mm256_load_ps(m_ptr + i), _mm256_load_ps(tns.m_ptr + i)));

        return *this;
    }

    // += value
    tensor &operator+=(const f32 value) {
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for (u32 i = 0; i < m_size; i += 8)
            _mm256_store_ps(m_ptr + i, _mm256_add_ps(_mm256_load_ps(m_ptr + i), val));

        return *this;
    }

    // -= value
    tensor &operator-=(const f32 value) {
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for (u32 i = 0; i < m_size; i += 8)
            _mm256_store_ps(m_ptr + i, _mm256_sub_ps(_mm256_load_ps(m_ptr + i), val));

        return *this;
    }

    // *= value
    tensor &operator*=(const f32 value) {
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for (u32 i = 0; i < m_size; i += 8)
            _mm256_store_ps(m_ptr + i, _mm256_mul_ps(_mm256_load_ps(m_ptr + i), val));

        return *this;
    }

    // +
    tensor operator+(const tensor &tns) {
        tensor nw(m_rows, m_cols, m_channels);
        for (u32 i = 0; i < m_size; i += 8)
            _mm256_store_ps(nw.m_ptr, _mm256_add_ps(_mm256_load_ps(m_ptr + i), _mm256_load_ps(tns.m_ptr + i)));

        return nw;
    }

    // -
    tensor operator-(const tensor &tns) {
        tensor nw(m_rows, m_cols, m_channels);
        for (u32 i = 0; i < m_size; i += 8)
            _mm256_store_ps(nw.m_ptr, _mm256_sub_ps(_mm256_load_ps(m_ptr + i), _mm256_load_ps(tns.m_ptr + i)));

        return nw;
    }

    // *
    tensor operator*(const tensor &tns) {
        tensor nw(m_rows, m_cols, m_channels);
        for (u32 i = 0; i < m_size; i += 8)
            _mm256_store_ps(nw.m_ptr, _mm256_mul_ps(_mm256_load_ps(m_ptr + i), _mm256_load_ps(tns.m_ptr + i)));

        return nw;
    }

    // + value
    tensor operator+(const f32 value) {
        tensor nw(m_rows, m_cols, m_channels);
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for (u32 i = 0; i < m_size; i += 8)
            _mm256_store_ps(nw.m_ptr + i, _mm256_add_ps(_mm256_load_ps(m_ptr + i), val));

        return nw;
    }

    // - value
    tensor operator-(const f32 value) {
        tensor nw(m_rows, m_cols, m_channels);
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for (u32 i = 0; i < m_size; i += 8)
            _mm256_store_ps(nw.m_ptr + i, _mm256_sub_ps(_mm256_load_ps(m_ptr + i), val));

        return nw;
    }

    // * value
    tensor operator*(const f32 value) {
        tensor nw(m_rows, m_cols, m_channels);
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for (u32 i = 0; i < m_size; i += 8)
            _mm256_store_ps(nw.m_ptr + i, _mm256_mul_ps(_mm256_load_ps(m_ptr + i), val));

        return nw;
    }

#elif defined(SSE) // ----- SSE OPERATOR OVERLOADING ----- //

    // += 
    tensor& operator+=(const tensor &tns) {
        for(u32 i = 0; i < m_size; i += 4)
            _mm_store_ps(m_ptr + i, _mm_add_ps(_mm_load_ps(m_ptr + i), _mm_load_ps(tns.m_ptr + i)));

        return *this;
    }

    // -= 
    tensor& operator-=(const tensor &tns) {
        for(u32 i = 0; i < m_size; i += 4)
            _mm_store_ps(m_ptr + i, _mm_sub_ps(_mm_load_ps(m_ptr + i), _mm_load_ps(tns.m_ptr + i)));

        return *this;
    }

    // *=
    tensor& operator*=(const tensor &tns) {
        for(u32 i = 0; i < m_size; i += 4)
            _mm_store_ps(m_ptr + i, _mm_mul_ps(_mm_load_ps(m_ptr + i), _mm_load_ps(tns.m_ptr + i)));

        return *this;
    }
   
    // += value
    tensor& operator+=(const f32 value) {
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < m_size; i += 4)
            _mm_store_ps(m_ptr + i, _mm_add_ps(_mm_load_ps(m_ptr + i), val));

        return *this;
    }

    // -= value
    tensor& operator-=(const f32 value) {
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < m_size; i += 4)
            _mm_store_ps(m_ptr + i, _mm_sub_ps(_mm_load_ps(m_ptr + i), val));

        return *this;
    }

    // *= value
    tensor& operator*=(const f32 value) {
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < m_size; i += 4)
            _mm_store_ps(m_ptr + i, _mm_mul_ps(_mm_load_ps(m_ptr + i), val));

        return *this;
    }

    // +
    tensor operator+(const tensor &tns) {
        tensor nw(m_rows, m_cols, m_channels);
        for(u32 i = 0; i < m_size; i += 4)
            _mm_store_ps(nw.m_ptr, _mm_add_ps(_mm_load_ps(m_ptr + i), _mm_load_ps(tns.m_ptr + i)));

        return nw;
    }

    // -
    tensor operator-(const tensor &tns) {
        tensor nw(m_rows, m_cols, m_channels);
        for(u32 i = 0; i < m_size; i += 4)
            _mm_store_ps(nw.m_ptr, _mm_sub_ps(_mm_load_ps(m_ptr + i), _mm_load_ps(tns.m_ptr + i)));
            
        return nw;
    }

    // *
    tensor operator*(const tensor &tns) {
        tensor nw(m_rows, m_cols, m_channels);
        for(u32 i = 0; i < m_size; i += 4)
            _mm_store_ps(nw.m_ptr, _mm_mul_ps(_mm_load_ps(m_ptr + i), _mm_load_ps(tns.m_ptr + i)));

        return nw;
    }
   
    // + value
    tensor operator+(const f32 value) {
        tensor nw(m_rows, m_cols, m_channels);
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < m_size; i += 4)
            _mm_store_ps(nw.m_ptr + i, _mm_add_ps(_mm_load_ps(m_ptr + i), val));

        return nw;
    }

    // - value
    tensor operator-(const f32 value) {
        tensor nw(m_rows, m_cols, m_channels);
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < m_size; i += 4)
            _mm_store_ps(nw.m_ptr + i, _mm_sub_ps(_mm_load_ps(m_ptr + i), val));

        return nw;
    }

    // * value
    tensor operator*(const f32 value) {
        tensor nw(m_rows, m_cols, m_channels);
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < m_size; i += 4)
            _mm_store_ps(nw.m_ptr + i, _mm_mul_ps(_mm_load_ps(m_ptr + i), val));

        return nw;
    }

#elif defined(NORMAL) // ----- NORMAL OPERATOR OVERLOADING ----- // 

    // += 
    tensor& operator+=(const tensor &tns) {
        for(u32 i = 0; i < m_size; i++) m_ptr[i] += tns.m_ptr[i];
        return *this;
    }

    // -= 
    tensor& operator-=(const tensor &tns) {
        for(u32 i = 0; i < m_size; i++) m_ptr[i] -= tns.m_ptr[i];
        return *this;
    }

    // *=
    tensor& operator*=(const tensor &tns) {
        for(u32 i = 0; i < m_size; i++) m_ptr[i] *= tns.m_ptr[i];
        return *this;
    }
   
    // += value
    tensor& operator+=(const f32 value) {
        for(u32 i = 0; i < m_size; i++) m_ptr[i] += value;
        return *this;
    }

    // -= value
    tensor& operator-=(const f32 value) {
        for(u32 i = 0; i < m_size; i++) m_ptr[i] -= value;
        return *this;
    }

    // *= value
    tensor& operator*=(const f32 value) {
        for(u32 i = 0; i < m_size; i++) m_ptr[i] *= value;
        return *this;
    }

    // +
    tensor operator+(const tensor &tns) {
        tensor nw(m_rows, m_cols, m_channels);
        for(u32 i = 0; i < m_size; i++) nw.m_ptr[i] = m_ptr[i] + tns.m_ptr[i];
        return nw;
    }

    // -
    tensor operator-(const tensor &tns) {
        tensor nw(m_rows, m_cols, m_channels);
        for(u32 i = 0; i < m_size; i++) nw.m_ptr[i] = m_ptr[i] - tns.m_ptr[i];
        return nw;
    }

    // *
    tensor operator*(const tensor &tns) {
        tensor nw(m_rows, m_cols, m_channels);
        for(u32 i = 0; i < m_size; i++) nw.m_ptr[i] = m_ptr[i] * tns.m_ptr[i];
        return nw;
    }
   
    // + value
    tensor operator+(const f32 value) {
        tensor nw(m_rows, m_cols, m_channels);
        for(u32 i = 0; i < m_size; i++) nw.m_ptr[i] = m_ptr[i] + value;
        return nw;
    }

    // - value
    tensor operator-(const f32 value) {
        tensor nw(m_rows, m_cols, m_channels);
        for(u32 i = 0; i < m_size; i++) nw.m_ptr[i] = m_ptr[i] - value;
        return nw;
    }

    // * value
    tensor operator*(const f32 value) {
        tensor nw(m_rows, m_cols, m_channels);
        for(u32 i = 0; i < m_size; i++) nw.m_ptr[i] = m_ptr[i] * value;
        return nw;
    }

#endif // ----- END OF OPERATOR OVERLOADING ----- //
};

} // namespace pinguml
