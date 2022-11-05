#pragma once

#include "../include.hpp"

namespace pinguml {

class tensor {
private:
    u32 _size;
    u32 _capacity;
    f32 *_memory;

    void _delete() { delete[] _memory; ptr = nullptr; _memory = nullptr; }

    #ifdef AVX
    f32 *_new(const u32 sz) { 
        _memory = new f32[sz + 31]; 
        ptr = (f32 *)(((std::uintptr_t)_memory + 32) & ~(std::uintptr_t)0x1F); 
        return ptr; 
    }
    #endif

    #ifdef SSE
    f32 *_new(const u32 sz) { 
        _memory = new f32[sz + 15]; 
        ptr = (f32 *)(((std::uintptr_t)_memory + 16) & ~(std::uintptr_t)0x0F); 
        return ptr; 
    }
    #endif

    #ifdef NORMAL
    f32 *_new(const u32 sz) { 
        _memory = new f32[sz]; 
        ptr = (f32 *)((std::uintptr_t)_memory);
        return ptr; 
    }
    #endif

public:
    u32 rows, cols, channels;
    u32 channel_stride;
    f32 *ptr;

    u32 size() { return _size; }

    u32 _channel_stride(const u32 h, const u32 w) {
        u32 x = h * w;

        #ifdef NORMAL 
        return x;
        #endif
        
        u8 chunk_size;

        #ifdef AVX
        chunk_size = 8;
        #endif

        #ifdef SSE
        chunk_size = 4;
        #endif

        const u32 rem = x % chunk_size;
        if(rem) x += (-rem) + chunk_size;
        return x;
    }

    tensor() : _size(0), _capacity(0), cols(0), rows(0), channels(0), channel_stride(0), ptr(0) {}
    
    tensor(const u32 h, const u32 w, const u32 c, const f32 *data = nullptr) : rows(h), cols(w), channels(c) {
        channel_stride = _channel_stride(h, w);
        _size = _capacity = channel_stride * channels; 
        ptr = _new(_size);
        if(data) std::memcpy(ptr, data, _size * sizeof(f32));
    }

    tensor(const tensor &tns) : _size(tns._size), _capacity(tns._capacity), rows(tns.rows), cols(tns.cols), channels(tns.channels), channel_stride(tns.channel_stride) {
        ptr = _new(_size);
        std::memcpy(ptr, tns.ptr, tns._size * sizeof(f32));
    }

    tensor(const tensor &tns, const u32 pad_rows, const u32 pad_cols, const u8 pad_type) : _size(tns._size), _capacity(tns._capacity), rows(tns.rows), cols(tns.cols), channels(tns.channels), channel_stride(tns.channel_stride) {
        ptr = _new(tns._size);
        std::memcpy(ptr, tns.ptr, tns._size * sizeof(f32));
        *this = tensor_pad(pad_rows, pad_cols, pad_rows, pad_cols, pad_type);
    }

    ~tensor() { _delete(); }

    tensor channel(const u32 index, const u32 nr_channels) const {
        return tensor(rows, cols, nr_channels, &ptr[channel_stride*index]);
    }


    tensor crop(const u32 dy, const u32 dx, const u32 h, const u32 w) const {
        tensor tns(h, w, channels);

        for(u32 i = 0; i < channels; i++) 
            for(u32 j = 0; j < h; j++) 
                std::memcpy(&tns.ptr[j * w + i * tns.channel_stride], &ptr[dx + (j + dy) * cols + channel_stride], w * sizeof(f32));

        return tns;
    }

    tensor shift(const i32 dy, const i32 dx, const u8 pad_type) {
        tensor shifted = tensor_pad(abs(dy), abs(dx), abs(dy), abs(dx), pad_type);
        return shifted.crop(abs(dy) - dy, abs(dx) - dx, rows, cols);
    }

    tensor flip_cols() {
        tensor tns(rows, cols, channels);

        for(u32 i = 0; i < channels; i++) 
            for(u32 j = 0; j < rows; j++) 
                for(u32 k = 0; k < cols; k++) 
                    tns.ptr[i * channel_stride + j * cols + k] = ptr[k * channel_stride + j * cols + (cols - k - 1)];
        
        return tns;
    }

    tensor flip_rows() {
        tensor tns(rows, cols, channels);

        for(u32 i = 0; i < channels; i++) 
            for(u32 j = 0; j < rows; j++) 
                std::memcpy(&tns.ptr[i * channel_stride + (rows - j - 1) * cols], &ptr[i * channel_stride + j * cols], cols * sizeof(f32));

        return tns;
    }

    f32 average_decrease() { 
        const u32 area = rows * cols;
        f32 average = 0;
        for(u32 i = 0; i < channels; i++) {
            const u32 channel_id = i * channel_stride;
            for(u32 j = channel_id; j < channel_id + area; j++) 
                average += ptr[j];
        }
        average /= f32(area * channels);

        for(u32 i = 0; i < _size; i++) ptr[i] -= average; // add SIMD
        return average;
    }

    f32 average_decrease(const u32 channel) {
        const u32 area = rows * cols;
        const u32 channel_id = channel * channel_stride;

        f32 average = 0;
        for(u32 i = channel_id; i < channel_id + area; i++) average += ptr[i]; // add SIMD

        average /= f32(area);
        for(u32 i = channel_id; i < channel_id + area; i++) ptr[i] -= average; // add SIMD

        return average;
    }

    void fill(const f32 value) { // add SIMD
        for(u32 i = 0; i < _size; i++) ptr[i] = value;
    }

    void fill_random_uniform(const f32 low, const f32 high) {
        std::mt19937 seed(0);
        std::uniform_real_distribution<f32> generator(low, high);
        for(u32 i = 0; i < _size; i++) ptr[i] = generator(seed);
    }
     
    void fill_random_normal(const f32 high) {
        std::mt19937 seed(0);
        std::normal_distribution<f32> generator(0, high);
        for(u32 i = 0; i < _size; i++) ptr[i] = generator(seed);
    }

    tensor tensor_pad(const u32 dy_top, const u32 dy_bottom, const u32 dx_left, const u32 dx_right, const u8 pad_type) const {
        tensor tns(rows + dy_top + dy_bottom, cols + dx_left + dx_right, channels);
        tns.fill(0);

        for(u32 i = 0; i < channels; i++) {
            const u32 tns_channel_id = i * tns.channel_stride;
            const u32 channel_id = i * channel_stride;

            f32 average = 0.f;
            if(pad_type == 2) {
                const u32 perimeter = 2 * (rows + cols) - 4;

                for(u32 j = 0; j < cols; j++) {
                    average += ptr[channel_id + j];
                    average += ptr[channel_id + cols * (rows-1) + j];
                }

                for(u32 j = 1; j < rows - 1; j++) {
                    average += ptr[channel_id + j * cols];
                    average = ptr[channel_id + j * cols + cols - 1];
                }

                average /= f32(perimeter);
            }

            // left and right padding (plus center)
            for(u32 j = 0; j < rows; j++) {
                std::memcpy(&tns.ptr[tns_channel_id + dx_left + (j + dy_top) * tns.cols], &ptr[channel_id + j * cols], cols * sizeof(f32));
                if(pad_type == 1) {
                    for(u32 k = 0; k < dx_left; k++) 
                        tns.ptr[tns_channel_id + (j + dy_top) * tns.cols + k] = ptr[channel_id + j * cols];
                    
                    for(u32 k = 0; k < dx_right; k++) 
                        tns.ptr[tns_channel_id + (j + dy_top) * tns.cols + k + dx_left + cols] = ptr[channel_id + j * cols + cols - 1];
                }
                else if(pad_type == 2) {
                    for(u32 k = 0; k < dx_left; k++) 
                        tns.ptr[tns_channel_id + (j + dy_top) * tns.cols + k] = average;

                    for(u32 k = 0; k < dx_right; k++) 
                        tns.ptr[tns_channel_id + (j + dy_top) * tns.cols + k + dx_left + cols] = average;
                }
            }

            // top and bottom padding
            if(pad_type == 1) {
                for(u32 j = 0; j < dy_top; j++) 
                    std::memcpy(&tns.ptr[tns_channel_id + j * tns.cols], &tns.ptr[tns_channel_id + dy_top * tns.cols], tns.cols * sizeof(f32));

                for(u32 j = 0; j < dy_bottom; j++) 
                    std::memcpy(&tns.ptr[tns_channel_id + (j + dy_top + rows) * tns.cols], &tns.ptr[tns_channel_id + (dy_top + rows - 1) * tns.cols], tns.cols * sizeof(f32));
            }
            else if(pad_type == 2) {
                for(u32 j = 0; j < dy_top; j++) 
                    for(u32 k = 0; k < tns.cols; k++) 
                        tns.ptr[tns_channel_id + tns.cols * j + k] = average;

                for(u32 j = 0; j < dy_bottom; j++) 
                    for(u32 k = 0; k < tns.cols; k++) 
                        tns.ptr[tns_channel_id + tns.cols * (j + dy_top + rows) + k] = average;
            }
        }

        return tns;
    }

    void resize(const u32 h, const u32 w, const u32 c) {
        const u32 new_stride = _channel_stride(h, w);
        const u32 sz = new_stride * c;
        if(sz > _capacity) {
            if(_capacity) _delete();
            _size = sz;
            _capacity = _size;
            ptr = _new(_size);
        }
        rows = h; 
        cols = w;
        channels = c;
        _size = sz;
        channel_stride = new_stride;
    }
    
    // = 
    tensor& operator=(const tensor &tns) {
        resize(tns.rows, tns.cols, tns.channels);
        std::memcpy(ptr, tns.ptr, _size * sizeof(f32));
        return *this;
    }

#if defined(AVX) // ----- AVX OPERATOR OVERLOADING ----- //

    // += 
    tensor& operator+=(const tensor &tns) {
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(ptr + i, _mm256_add_ps(_mm256_load_ps(ptr + i), _mm256_load_ps(tns.ptr + i)));

        return *this;
    }

    // -= 
    tensor& operator-=(const tensor &tns) {
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(ptr + i, _mm256_sub_ps(_mm256_load_ps(ptr + i), _mm256_load_ps(tns.ptr + i)));

        return *this;
    }

    // *=
    tensor& operator*=(const tensor &tns) {
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(ptr + i, _mm256_mul_ps(_mm256_load_ps(ptr + i), _mm256_load_ps(tns.ptr + i)));

        return *this;
    }
   
    // += value
    tensor& operator+=(const f32 value) {
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(u32 i = 0; i < _size; i += 8)
            _mm256_store_ps(ptr + i, _mm256_add_ps(_mm256_load_ps(ptr + i), val));

        return *this;
    }

    // -= value
    tensor& operator-=(const f32 value) {
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(ptr + i, _mm256_sub_ps(_mm256_load_ps(ptr + i), val));

        return *this;
    }

    // *= value
    tensor& operator*=(const f32 value) {
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(ptr + i, _mm256_mul_ps(_mm256_load_ps(ptr + i), val));

        return *this;
    }

    // +
    tensor operator+(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw.ptr, _mm256_add_ps(_mm256_load_ps(ptr + i), _mm256_load_ps(tns.ptr + i)));

        return nw;
    }

    // -
    tensor operator-(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw.ptr, _mm256_sub_ps(_mm256_load_ps(ptr + i), _mm256_load_ps(tns.ptr + i)));
            
        return nw;
    }

    // *
    tensor operator*(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw.ptr, _mm256_mul_ps(_mm256_load_ps(ptr + i), _mm256_load_ps(tns.ptr + i)));

        return nw;
    }
   
    // + value
    tensor operator+(const f32 value) {
        tensor nw(rows, cols, channels);
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw.ptr + i, _mm256_add_ps(_mm256_load_ps(ptr + i), val));

        return nw;
    }

    // - value
    tensor operator-(const f32 value) {
        tensor nw(rows, cols, channels);
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw.ptr + i, _mm256_sub_ps(_mm256_load_ps(ptr + i), val));

        return nw;
    }

    // * value
    tensor operator*(const f32 value) {
        tensor nw(rows, cols, channels);
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(u32 i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw.ptr + i, _mm256_mul_ps(_mm256_load_ps(ptr + i), val));

        return nw;
    }

#elif defined(SSE) // ----- SSE OPERATOR OVERLOADING ----- //
    
    // += 
    tensor& operator+=(const tensor &tns) {
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(ptr + i, _mm_add_ps(_mm_load_ps(ptr + i), _mm_load_ps(tns.ptr + i)));

        return *this;
    }

    // -= 
    tensor& operator-=(const tensor &tns) {
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(ptr + i, _mm_sub_ps(_mm_load_ps(ptr + i), _mm_load_ps(tns.ptr + i)));

        return *this;
    }

    // *=
    tensor& operator*=(const tensor &tns) {
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(ptr + i, _mm_mul_ps(_mm_load_ps(ptr + i), _mm_load_ps(tns.ptr + i)));

        return *this;
    }
   
    // += value
    tensor& operator+=(const f32 value) {
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < _size; i += 4)
            _mm_store_ps(ptr + i, _mm_add_ps(_mm_load_ps(ptr + i), val));

        return *this;
    }

    // -= value
    tensor& operator-=(const f32 value) {
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(ptr + i, _mm_sub_ps(_mm_load_ps(ptr + i), val));

        return *this;
    }

    // *= value
    tensor& operator*=(const f32 value) {
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(ptr + i, _mm_mul_ps(_mm_load_ps(ptr + i), val));

        return *this;
    }

    // +
    tensor operator+(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(nw.ptr, _mm_add_ps(_mm_load_ps(ptr + i), _mm_load_ps(tns.ptr + i)));

        return nw;
    }

    // -
    tensor operator-(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(nw.ptr, _mm_sub_ps(_mm_load_ps(ptr + i), _mm_load_ps(tns.ptr + i)));
            
        return nw;
    }

    // *
    tensor operator*(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(nw.ptr, _mm_mul_ps(_mm_load_ps(ptr + i), _mm_load_ps(tns.ptr + i)));

        return nw;
    }
   
    // + value
    tensor operator+(const f32 value) {
        tensor nw(rows, cols, channels);
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(nw.ptr + i, _mm_add_ps(_mm_load_ps(ptr + i), val));

        return nw;
    }

    // - value
    tensor operator-(const f32 value) {
        tensor nw(rows, cols, channels);
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(nw.ptr + i, _mm_sub_ps(_mm_load_ps(ptr + i), val));

        return nw;
    }

    // * value
    tensor operator*(const f32 value) {
        tensor nw(rows, cols, channels);
        __m128 val = _mm_set_ps(value, value, value, value);
        for(u32 i = 0; i < _size; i += 4) 
            _mm_store_ps(nw.ptr + i, _mm_mul_ps(_mm_load_ps(ptr + i), val));

        return nw;
    }

#elif defined(NORMAL) // ----- NORMAL OPERATOR OVERLOADING ----- // 

    // += 
    tensor& operator+=(const tensor &tns) {
        for(u32 i = 0; i < _size; i++) ptr[i] += tns.ptr[i];
        return *this;
    }

    // -= 
    tensor& operator-=(const tensor &tns) {
        for(u32 i = 0; i < _size; i++) ptr[i] -= tns.ptr[i];
        return *this;
    }

    // *=
    tensor& operator*=(const tensor &tns) {
        for(u32 i = 0; i < _size; i++) ptr[i] *= tns.ptr[i];
        return *this;
    }
   
    // += value
    tensor& operator+=(const f32 value) {
        for(u32 i = 0; i < _size; i++) ptr[i] += value;
        return *this;
    }

    // -= value
    tensor& operator-=(const f32 value) {
        for(u32 i = 0; i < _size; i++) ptr[i] -= value;
        return *this;
    }

    // *= value
    tensor& operator*=(const f32 value) {
        for(u32 i = 0; i < _size; i++) ptr[i] *= value;
        return *this;
    }

    // +
    tensor operator+(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i++) nw.ptr[i] = ptr[i] + tns.ptr[i];
        return nw;
    }

    // -
    tensor operator-(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i++) nw.ptr[i] = ptr[i] - tns.ptr[i];
        return nw;
    }

    // *
    tensor operator*(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i++) nw.ptr[i] = ptr[i] * tns.ptr[i];
        return nw;
    }
   
    // + value
    tensor operator+(const f32 value) {
        tensor nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i++) nw.ptr[i] = ptr[i] + value;
        return nw;
    }

    // - value
    tensor operator-(const f32 value) {
        tensor nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i++) nw.ptr[i] = ptr[i] - value;
        return nw;
    }

    // * value
    tensor operator*(const f32 value) {
        tensor nw(rows, cols, channels);
        for(u32 i = 0; i < _size; i++) nw.ptr[i] = ptr[i] * value;
        return nw;
    }

#endif // ----- END OF OPERATOR OVERLOADING ----- //
};

} // namespace pinguml
