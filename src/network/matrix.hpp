#pragma once

#include <random>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>

#include "../defines.hpp"

namespace pingu {

enum padding { zero = 0, edge = 1, median = 2 };

class matrix {
    u32 _size;
    u32 _capacity;
    f32 *_memory;

    void _delete() { delete[] _memory; _ptr = 0; _memory = 0; }

#ifdef AVX
    f32 *_new(const u32 sz) { 
        _memory = new f32[sz + 31]; 
        _ptr = (f32 *)(((std::uintptr_t)_memory + 32) & ~(std::uintptr_t)0x1F); 
        return _ptr; 
    }
#endif

#ifdef SSE
    f32 *_new(const u32 sz) { 
        _memory = new f32[sz + 15]; 
        _ptr = (f32 *)(((std::uintptr_t)_memory + 16) & ~(std::uintptr_t)0x0F); 
        return _ptr; 
    }
#endif

#ifdef NORMAL
    f32 *_new(const u32 sz) { 
        _memory = new f32[sz]; // 32 byte alignment
        _ptr = (f32 *)((std::uintptr_t)_memory);
        return _ptr; 
    }
#endif

public:
    u32 rows, cols, channels;
    u32 channel_stride;
    u32 channels_aligned;
    f32 *_ptr;

    u32 size() { return _size; }

#ifdef AVX
    u32 _channel_stride(const u32 h, const u32 w) {
        u32 x = h * w;
        if(channels_aligned) {
            const u32 rem = x % 8;
            if(rem) x += (-rem) + 8;
            return x;
        }
        return x; 
    }
#endif

#ifdef SSE 
    u32 _channel_stride(const u32 h, const u32 w) { 
        u32 x = h * w;
        if(channels_aligned) {
            const u32 rem = x % 4;
            if(rem) x += (-rem) + 4;
            return x;
        }
        return x; 
    }
#endif

#ifdef NORMAL 
    u32 _channel_stride(const u32 h, const u32 w) { 
        return h * w;
    }
#endif

    matrix() : _size(0), _capacity(0), cols(0), rows(0), channels(0), channel_stride(0), channels_aligned(0), _ptr(0) {}
      
    matrix(const u32 h, const u32 w, const u32 c = 1, const f32 *data = 0, const u32 align_channels = 0) : rows(h), cols(w), channels(c) {
        channels_aligned = align_channels;
        channel_stride = _channel_stride(h, w);
        _size = _capacity = channel_stride * channels; 
        _ptr = _new(_size);
        if(data) std::memcpy(_ptr, data, _size * sizeof(f32));
    }

    matrix(const matrix &mtx) : _size(mtx._size), _capacity(mtx._capacity), rows(mtx.rows), cols(mtx.cols), channels(mtx.channels), channel_stride(mtx.channel_stride), channels_aligned(mtx.channels_aligned) {
        _ptr = _new(_size);
        std::memcpy(_ptr, mtx._ptr, mtx._size * sizeof(f32));
    }

    matrix(const matrix &mtx, const u32 pad_rows, const u32 pad_cols, pingu::padding pad_type = pingu::zero, const u32 threads = 1) : _size(mtx._size), _capacity(mtx._capacity), rows(mtx.rows), cols(mtx.cols), channels(mtx.channels), channel_stride(mtx.channel_stride), channels_aligned(mtx.channels_aligned) {
        _ptr = _new(mtx._size);
        std::memcpy(_ptr, mtx._ptr, mtx._size * sizeof(f32));
        *this = matrix_pad(pad_rows, pad_cols, pad_type, threads);
    }

    ~matrix() { if(_ptr) _delete(); }

    matrix channel(const u32 index, const u32 nr_channels = 1) const {
        return matrix(rows, cols, nr_channels, &_ptr[channel_stride*index]);
    }


    matrix crop(const u32 dy, const u32 dx, const u32 h, const u32 w, const u32 threads = 1) const {
        matrix mtx(h, w, channels);

        for(u32 i = 0; i < channels; i++) 
            for(u32 j = 0; j < h; j++) 
                std::memcpy(&mtx._ptr[j * w + i * mtx.channel_stride], &_ptr[dx + (j + dy) * cols + channel_stride], w * sizeof(f32));

        return mtx;
    }

    matrix shift(const i32 dy, const i32 dx, pingu::padding pad_type = pingu::zero) {
        matrix shifted = matrix_pad(abs(dy), abs(dx), pad_type);
        return shifted.crop(abs(dy) - dy, abs(dx) - dx, rows, cols);
    }

    matrix flip_cols() {
        matrix mtx(rows, cols, channels);

        for(u32 i = 0; i < channels; i++) 
            for(u32 j = 0; j < rows; j++) 
                for(u32 k = 0; k < cols; k++) 
                    mtx._ptr[i * channel_stride + j * cols + k] = _ptr[k * channel_stride + j * cols + (cols - k -1)];
        
        return mtx;
    }

    matrix flip_rows() {
        matrix mtx(rows, cols, channels);

        for(u32 i = 0; i < channels; i++) 
            for(u32 j = 0; j < rows; j++) 
                std::memcpy(&mtx._ptr[i * channel_stride + (rows - j - 1) * cols], &_ptr[i * channel_stride + j * cols], cols * sizeof(f32));

        return mtx;
    }

    f32 mean() {
        const u32 area = rows * cols;
        f32 average = 0;
        for(u32 i = 0; i < channels; i++) {
            const u32 channel_id = i * channel_stride;
            for(u32 j = channel_id; j < channel_id + area; j++) 
                average += _ptr[j];
        }

        average /= f32(area * channels);
        return average;
    }

    f32 mean_decrease() {
        const f32 dm = mean();
        for(u32 i = 0; i < _size; i++) _ptr[i] -= dm;
        return dm;
    }

    f32 mean_decrease(const u32 channel) {
        const u32 area = rows * cols;
        const u32 channel_id = channel * channel_stride;

        f32 average = 0;
        for(u32 i = channel_id; i < channel_id + area; i++) average += _ptr[i];

        average /= f32(area);
        for(u32 i = channel_id; i < channel_id + area; i++) _ptr[i] -= average;

        return average;
    }

    void fill(const f32 value) { 
        for(u32 i = 0; i < _size; i++) _ptr[i] = value;
    }

    void fill_random(const f32 low, const f32 high) {
        std::mt19937 seed(0);
        std::uniform_real_distribution<f32> generator(low, high);
        for(u32 i = 0; i < _size; i++) _ptr[i] = generator(seed);
    }

    matrix matrix_pad(const u32 dy_top, const u32 dy_bottom, const u32 dx_left, const u32 dx_right, pingu::padding pad_type = pingu::zero, const u32 threads = 1) const {
        matrix mtx(rows + dy_top + dy_bottom, cols + dx_left + dx_right, channels);
        mtx.fill(0);
 
        for(u32 i = 0; i < channels; i++) {
            const u32 mtx_channel_id = i * mtx.channel_stride;
            const u32 channel_id = i * channel_stride;

            f32 median = 0.f;
            if(pad_type == pingu::median) {
                const u32 perimeter = 2 * (rows + cols) - 4;
                std::vector<f32> border(perimeter);

                for(u32 j = 0; j < cols; j++) {
                    border[j] = _ptr[channel_id + j];
                    border[cols + j] = _ptr[channel_id + cols * (rows-1) + j];
                }

                for(u32 j = 1; j < rows - 1; j++) {
                    border[cols * 2 + j - 1] = _ptr[channel_id + j * cols];
                    border[perimeter - j] = _ptr[channel_id + j * cols + cols - 1];
                }

                std::nth_element(border.begin(), border.begin() + perimeter / 2, border.end());
                median = border[perimeter / 2];
            }

            // left and right padding (plus center)
            for(u32 j = 0; j < rows; j++) {
                std::memcpy(&mtx._ptr[mtx_channel_id + dx_left + (j + dy_top) * mtx.cols], &_ptr[channel_id + j * cols], cols * sizeof(f32));
                if(pad_type == pingu::edge) {
                    for(u32 k = 0; k < dx_left; k++) 
                        mtx._ptr[mtx_channel_id + (j + dy_top) * mtx.cols + k] = _ptr[channel_id + j * cols];
                    
                    for(u32 k = 0; k < dx_right; k++) 
                        mtx._ptr[mtx_channel_id + (j + dy_top) * mtx.cols + k + dx_left + cols] = _ptr[channel_id + j * cols + cols - 1];
                }
                else if(pad_type == pingu::median) {
                    for(u32 k = 0; k < dx_left; k++) 
                        mtx._ptr[mtx_channel_id + (j + dy_top) * mtx.cols + k] = median;

                    for(u32 k = 0; k < dx_right; k++) 
                        mtx._ptr[mtx_channel_id + (j + dy_top) * mtx.cols + k + dx_left + cols] = median;
                }
            }

            // top and bottom padding
            if(pad_type == pingu::edge) {
                for(u32 j = 0; j < dy_top; j++) 
                    std::memcpy(&mtx._ptr[mtx_channel_id + j * mtx.cols], &mtx._ptr[mtx_channel_id + dy_top * mtx.cols], mtx.cols * sizeof(f32));

                for(u32 j = 0; j < dy_bottom; j++) 
                    std::memcpy(&mtx._ptr[mtx_channel_id + (j + dy_top + rows) * mtx.cols], &mtx._ptr[mtx_channel_id + (dy_top + rows - 1) * mtx.cols], mtx.cols * sizeof(f32));
            }
            else if(pad_type == pingu::median) {
                for(u32 j = 0; j < dy_top; j++) 
                    for(u32 k = 0; k < mtx.cols; k++) 
                        mtx._ptr[mtx_channel_id + mtx.cols * j + k] = median;

                for(u32 j = 0; j < dy_bottom; j++) 
                    for(u32 k = 0; k < mtx.cols; k++) 
                        mtx._ptr[mtx_channel_id + mtx.cols * (j + dy_top + rows) + k] = median;
            }
        }

        return mtx;
    }

    matrix matrix_pad(const u32 dy, const u32 dx, pingu::padding pad_type = pingu::zero, const u32 threads = 1) const {
        return matrix_pad(dy, dx, dy, dx, pad_type, threads);
    }

    void resize(const u32 h, const u32 w, const u32 c, const u32 align_channels = 0) {
        channels_aligned = align_channels;
        const u32 new_stride = _channel_stride(h, w);
        const u32 sz = new_stride * c;
        if(sz > _capacity) {
            if(_capacity) _delete();
            _size = sz;
            _capacity = _size;
            _ptr = _new(_size);
        }
        rows = h; 
        cols = w;
        channels = c;
        _size = sz;
        channel_stride = new_stride;
    }
    
    // = 
    matrix& operator=(const matrix &mtx) {
        resize(mtx.rows, mtx.cols, mtx.channels, mtx.channels_aligned);
        std::memcpy(_ptr, mtx._ptr, _size * sizeof(f32));
        return *this;
    }

#ifdef AVX
    // += 
    matrix& operator+=(const matrix &mtx) {
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(_ptr + i, _mm256_add_ps(_mm256_load_ps(_ptr + i), _mm256_load_ps(mtx._ptr + i)));

        return *this;
    }

    // -= 
    matrix& operator-=(const matrix &mtx) {
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(_ptr + i, _mm256_sub_ps(_mm256_load_ps(_ptr + i), _mm256_load_ps(mtx._ptr + i)));

        return *this;
    }

    // *=
    matrix& operator*=(const matrix &mtx) {
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(_ptr + i, _mm256_mul_ps(_mm256_load_ps(_ptr + i), _mm256_load_ps(mtx._ptr + i)));

        return *this;
    }
   
    // += value
    matrix& operator+=(const f32 value) {
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(u32 i = 0; i < _size; i += 8)
            _mm256_store_ps(_ptr + i, _mm256_add_ps(_mm256_load_ps(_ptr + i), val));

        return *this;
    }

    // -= value
    matrix& operator-=(const f32 value) {
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(_ptr + i, _mm256_sub_ps(_mm256_load_ps(_ptr + i), val));

        return *this;
    }

    // *= value
    matrix& operator*=(const f32 value) {
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(_ptr + i, _mm256_mul_ps(_mm256_load_ps(_ptr + i), val));

        return *this;
    }

    // +
    matrix operator+(const matrix &mtx) {
        matrix nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw._ptr, _mm256_add_ps(_mm256_load_ps(_ptr + i), _mm256_load_ps(mtx._ptr + i)));

        return nw;
    }

    // -
    matrix operator-(const matrix &mtx) {
        matrix nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw._ptr, _mm256_sub_ps(_mm256_load_ps(_ptr + i), _mm256_load_ps(mtx._ptr + i)));
            
        return nw;
    }

    // *
    matrix operator*(const matrix &mtx) {
        matrix nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw._ptr, _mm256_mul_ps(_mm256_load_ps(_ptr + i), _mm256_load_ps(mtx._ptr + i)));

        return nw;
    }
   
    // + value
    matrix operator+(const f32 value) {
        matrix nw(rows, cols, channels);
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw._ptr + i, _mm256_add_ps(_mm256_load_ps(_ptr + i), val));

        return nw;
    }

    // - value
    matrix operator-(const f32 value) {
        matrix nw(rows, cols, channels);
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw._ptr + i, _mm256_sub_ps(_mm256_load_ps(_ptr + i), val));

        return nw;
    }

    // * value
    matrix operator*(const f32 value) {
        matrix nw(rows, cols, channels);
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw._ptr + i, _mm256_mul_ps(_mm256_load_ps(_ptr + i), val));

        return nw;
    }
#endif

#ifdef SSE
    // += 
    matrix& operator+=(const matrix &mtx) {
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(_ptr + i, _mm_add_ps(_mm_load_ps(_ptr + i), _mm_load_ps(mtx._ptr + i)));

        return *this;
    }

    // -= 
    matrix& operator-=(const matrix &mtx) {
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(_ptr + i, _mm_sub_ps(_mm_load_ps(_ptr + i), _mm_load_ps(mtx._ptr + i)));

        return *this;
    }

    // *=
    matrix& operator*=(const matrix &mtx) {
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(_ptr + i, _mm_mul_ps(_mm_load_ps(_ptr + i), _mm_load_ps(mtx._ptr + i)));

        return *this;
    }
   
    // += value
    matrix& operator+=(const f32 value) {
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < _size; i += 4)
            _mm_store_ps(_ptr + i, _mm_add_ps(_mm_load_ps(_ptr + i), val));

        return *this;
    }

    // -= value
    matrix& operator-=(const f32 value) {
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(_ptr + i, _mm_sub_ps(_mm_load_ps(_ptr + i), val));

        return *this;
    }

    // *= value
    matrix& operator*=(const f32 value) {
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(_ptr + i, _mm_mul_ps(_mm_load_ps(_ptr + i), val));

        return *this;
    }

    // +
    matrix operator+(const matrix &mtx) {
        matrix nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(nw._ptr, _mm_add_ps(_mm_load_ps(_ptr + i), _mm_load_ps(mtx._ptr + i)));

        return nw;
    }

    // -
    matrix operator-(const matrix &mtx) {
        matrix nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(nw._ptr, _mm_sub_ps(_mm_load_ps(_ptr + i), _mm_load_ps(mtx._ptr + i)));
            
        return nw;
    }

    // *
    matrix operator*(const matrix &mtx) {
        matrix nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(nw._ptr, _mm_mul_ps(_mm_load_ps(_ptr + i), _mm_load_ps(mtx._ptr + i)));

        return nw;
    }
   
    // + value
    matrix operator+(const f32 value) {
        matrix nw(rows, cols, channels);
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(nw._ptr + i, _mm_add_ps(_mm_load_ps(_ptr + i), val));

        return nw;
    }

    // - value
    matrix operator-(const f32 value) {
        matrix nw(rows, cols, channels);
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(nw._ptr + i, _mm_sub_ps(_mm_load_ps(_ptr + i), val));

        return nw;
    }

    // * value
    matrix operator*(const f32 value) {
        matrix nw(rows, cols, channels);
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(nw._ptr + i, _mm_mul_ps(_mm_load_ps(_ptr + i), val));

        return nw;
    }
#endif

#ifdef NORMAL
    // += 
    matrix& operator+=(const matrix &mtx) {
        for(u32 i = 0; i < _size; i++) _ptr[i] += mtx._ptr[i];
        return *this;
    }

    // -= 
    matrix& operator-=(const matrix &mtx) {
        for(u32 i = 0; i < _size; i++) _ptr[i] -= mtx._ptr[i];
        return *this;
    }

    // *=
    matrix& operator*=(const matrix &mtx) {
        for(u32 i = 0; i < _size; i++) _ptr[i] *= mtx._ptr[i];
        return *this;
    }
   
    // += value
    matrix& operator+=(const f32 value) {
        for(u32 i = 0; i < _size; i++) _ptr[i] += value;
        return *this;
    }

    // -= value
    matrix& operator-=(const f32 value) {
        for(u32 i = 0; i < _size; i++) _ptr[i] -= value;
        return *this;
    }

    // *= value
    matrix& operator*=(const f32 value) {
        for(u32 i = 0; i < _size; i++) _ptr[i] *= value;
        return *this;
    }

    // +
    matrix operator+(const matrix &mtx) {
        matrix nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i++) nw._ptr[i] = _ptr[i] + mtx._ptr[i];
        return nw;
    }

    // -
    matrix operator-(const matrix &mtx) {
        matrix nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i++) nw._ptr[i] = _ptr[i] - mtx._ptr[i];
        return nw;
    }

    // *
    matrix operator*(const matrix &mtx) {
        matrix nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i++) nw._ptr[i] = _ptr[i] * mtx._ptr[i];
        return nw;
    }
   
    // + value
    matrix operator+(const f32 value) {
        matrix nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i++) nw._ptr[i] = _ptr[i] + value;
        return nw;
    }

    // - value
    matrix operator-(const f32 value) {
        matrix nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i++) nw._ptr[i] = _ptr[i] - value;
        return nw;
    }

    // * value
    matrix operator*(const f32 value) {
        matrix nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i++) nw._ptr[i] = _ptr[i] * value;
        return nw;
    }
#endif

};

} // namespace pingu
