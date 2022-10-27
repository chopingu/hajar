#pragma once

#include <random>
#include <vector>
#include <cstring>
#include <cstdint>

#include "../defines.hpp"

namespace pinguml {

class tensor {
private:
    uint32_t _size;
    uint32_t _capacity;
    float *_memory;

    void _delete() { delete[] _memory; _ptr = 0; _memory = 0; }

    #ifdef AVX
    float *_new(const uint32_t sz) { 
        _memory = new float[sz + 31]; 
        _ptr = (float *)(((std::uintptr_t)_memory + 32) & ~(std::uintptr_t)0x1F); 
        return _ptr; 
    }
    #endif

    #ifdef SSE
    float *_new(const uint32_t sz) { 
        _memory = new float[sz + 15]; 
        _ptr = (float *)(((std::uintptr_t)_memory + 16) & ~(std::uintptr_t)0x0F); 
        return _ptr; 
    }
    #endif

    #ifdef NORMAL
    float *_new(const uint32_t sz) { 
        _memory = new float[sz]; 
        _ptr = (float *)((std::uintptr_t)_memory);
        return _ptr; 
    }
    #endif

public:
    uint32_t rows, cols, channels;
    uint32_t channel_stride;
    float *_ptr;

    uint32_t size() { return _size; }

    uint32_t _channel_stride(const uint32_t h, const uint32_t w) {
        uint32_t x = h * w;

        #ifdef NORMAL 
        return x;
        #endif
        
        uint8_t chunk_size;

        #ifdef AVX
        chunk_size = 8;
        #endif

        #ifdef SSE
        chunk_size = 4;
        #endif

        const uint32_t rem = x % chunk_size;
        if(rem) x += (-rem) + chunk_size;
        return x;
    }

    tensor() : _size(0), _capacity(0), cols(0), rows(0), channels(0), channel_stride(0), _ptr(0) {}
    
    tensor(const uint32_t h, const uint32_t w, const uint32_t c, const float *data = 0) : rows(h), cols(w), channels(c) {
        channel_stride = _channel_stride(h, w);
        _size = _capacity = channel_stride * channels; 
        _ptr = _new(_size);
        if(data) std::memcpy(_ptr, data, _size * sizeof(float));
    }

    tensor(const tensor &tns) : _size(tns._size), _capacity(tns._capacity), rows(tns.rows), cols(tns.cols), channels(tns.channels), channel_stride(tns.channel_stride) {
        _ptr = _new(_size);
        std::memcpy(_ptr, tns._ptr, tns._size * sizeof(float));
    }

    tensor(const tensor &tns, const uint32_t pad_rows, const uint32_t pad_cols, const uint8_t pad_type) : _size(tns._size), _capacity(tns._capacity), rows(tns.rows), cols(tns.cols), channels(tns.channels), channel_stride(tns.channel_stride) {
        _ptr = _new(tns._size);
        std::memcpy(_ptr, tns._ptr, tns._size * sizeof(float));
        *this = tensor_pad(pad_rows, pad_cols, pad_rows, pad_cols, pad_type);
    }

    ~tensor() { if(_ptr) _delete(); }

    tensor channel(const uint32_t index, const uint32_t nr_channels) const {
        return tensor(rows, cols, nr_channels, &_ptr[channel_stride*index]);
    }


    tensor crop(const uint32_t dy, const uint32_t dx, const uint32_t h, const uint32_t w) const {
        tensor tns(h, w, channels);

        for(uint32_t i = 0; i < channels; i++) 
            for(uint32_t j = 0; j < h; j++) 
                std::memcpy(&tns._ptr[j * w + i * tns.channel_stride], &_ptr[dx + (j + dy) * cols + channel_stride], w * sizeof(float));

        return tns;
    }

    tensor shift(const int32_t dy, const int32_t dx, const uint8_t pad_type) {
        tensor shifted = tensor_pad(abs(dy), abs(dx), abs(dy), abs(dx), pad_type);
        return shifted.crop(abs(dy) - dy, abs(dx) - dx, rows, cols);
    }

    tensor flip_cols() {
        tensor tns(rows, cols, channels);

        for(uint32_t i = 0; i < channels; i++) 
            for(uint32_t j = 0; j < rows; j++) 
                for(uint32_t k = 0; k < cols; k++) 
                    tns._ptr[i * channel_stride + j * cols + k] = _ptr[k * channel_stride + j * cols + (cols - k -1)];
        
        return tns;
    }

    tensor flip_rows() {
        tensor tns(rows, cols, channels);

        for(uint32_t i = 0; i < channels; i++) 
            for(uint32_t j = 0; j < rows; j++) 
                std::memcpy(&tns._ptr[i * channel_stride + (rows - j - 1) * cols], &_ptr[i * channel_stride + j * cols], cols * sizeof(float));

        return tns;
    }

    float average_decrease() { 
        const uint32_t area = rows * cols;
        float average = 0;
        for(uint32_t i = 0; i < channels; i++) {
            const uint32_t channel_id = i * channel_stride;
            for(uint32_t j = channel_id; j < channel_id + area; j++) 
                average += _ptr[j];
        }
        average /= float(area * channels);

        for(uint32_t i = 0; i < _size; i++) _ptr[i] -= average; // add SIMD
        return average;
    }

    float average_decrease(const uint32_t channel) {
        const uint32_t area = rows * cols;
        const uint32_t channel_id = channel * channel_stride;

        float average = 0;
        for(uint32_t i = channel_id; i < channel_id + area; i++) average += _ptr[i]; // add SIMD

        average /= float(area);
        for(uint32_t i = channel_id; i < channel_id + area; i++) _ptr[i] -= average; // add SIMD

        return average;
    }

    void fill(const float value) { // add SIMD
        for(uint32_t i = 0; i < _size; i++) _ptr[i] = value;
    }

    void fill_random_uniform(const float low, const float high) {
        std::mt19937 seed(0);
        std::uniform_real_distribution<float> generator(low, high);
        for(uint32_t i = 0; i < _size; i++) _ptr[i] = generator(seed);
    }
     
    void fill_random_normal(const float high) {
        std::mt19937 seed(0);
        std::normal_distribution<float> generator(0, high);
        for(uint32_t i = 0; i < _size; i++) _ptr[i] = generator(seed);
    }

    tensor tensor_pad(const uint32_t dy_top, const uint32_t dy_bottom, const uint32_t dx_left, const uint32_t dx_right, const uint8_t pad_type) const {
        tensor tns(rows + dy_top + dy_bottom, cols + dx_left + dx_right, channels);
        tns.fill(0);

        for(uint32_t i = 0; i < channels; i++) {
            const uint32_t tns_channel_id = i * tns.channel_stride;
            const uint32_t channel_id = i * channel_stride;

            float median = 0.f;
            if(!pad_type) {
                const uint32_t perimeter = 2 * (rows + cols) - 4;
                std::vector<float> border(perimeter);

                for(uint32_t j = 0; j < cols; j++) {
                    border[j] = _ptr[channel_id + j];
                    border[cols + j] = _ptr[channel_id + cols * (rows-1) + j];
                }

                for(uint32_t j = 1; j < rows - 1; j++) {
                    border[cols * 2 + j - 1] = _ptr[channel_id + j * cols];
                    border[perimeter - j] = _ptr[channel_id + j * cols + cols - 1];
                }

                std::nth_element(border.begin(), border.begin() + perimeter / 2, border.end());
                median = border[perimeter / 2];
            }

            // left and right padding (plus center)
            for(uint32_t j = 0; j < rows; j++) {
                std::memcpy(&tns._ptr[tns_channel_id + dx_left + (j + dy_top) * tns.cols], &_ptr[channel_id + j * cols], cols * sizeof(float));
                if(pad_type == 1) {
                    for(uint32_t k = 0; k < dx_left; k++) 
                        tns._ptr[tns_channel_id + (j + dy_top) * tns.cols + k] = _ptr[channel_id + j * cols];
                    
                    for(uint32_t k = 0; k < dx_right; k++) 
                        tns._ptr[tns_channel_id + (j + dy_top) * tns.cols + k + dx_left + cols] = _ptr[channel_id + j * cols + cols - 1];
                }
                else if(pad_type == 2) {
                    for(uint32_t k = 0; k < dx_left; k++) 
                        tns._ptr[tns_channel_id + (j + dy_top) * tns.cols + k] = median;

                    for(uint32_t k = 0; k < dx_right; k++) 
                        tns._ptr[tns_channel_id + (j + dy_top) * tns.cols + k + dx_left + cols] = median;
                }
            }

            // top and bottom padding
            if(pad_type == 1) {
                for(uint32_t j = 0; j < dy_top; j++) 
                    std::memcpy(&tns._ptr[tns_channel_id + j * tns.cols], &tns._ptr[tns_channel_id + dy_top * tns.cols], tns.cols * sizeof(float));

                for(uint32_t j = 0; j < dy_bottom; j++) 
                    std::memcpy(&tns._ptr[tns_channel_id + (j + dy_top + rows) * tns.cols], &tns._ptr[tns_channel_id + (dy_top + rows - 1) * tns.cols], tns.cols * sizeof(float));
            }
            else if(pad_type == 2) {
                for(uint32_t j = 0; j < dy_top; j++) 
                    for(uint32_t k = 0; k < tns.cols; k++) 
                        tns._ptr[tns_channel_id + tns.cols * j + k] = median;

                for(uint32_t j = 0; j < dy_bottom; j++) 
                    for(uint32_t k = 0; k < tns.cols; k++) 
                        tns._ptr[tns_channel_id + tns.cols * (j + dy_top + rows) + k] = median;
            }
        }

        return tns;
    }

    void resize(const uint32_t h, const uint32_t w, const uint32_t c) {
        const uint32_t new_stride = _channel_stride(h, w);
        const uint32_t sz = new_stride * c;
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
    tensor& operator=(const tensor &tns) {
        resize(tns.rows, tns.cols, tns.channels);
        std::memcpy(_ptr, tns._ptr, _size * sizeof(float));
        return *this;
    }

#if defined(AVX) // ----- AVX OPERATOR OVERLOADING ----- //

    // += 
    tensor& operator+=(const tensor &tns) {
        for(uint32_t i = 0; i < _size; i += 8) 
            _mm256_store_ps(_ptr + i, _mm256_add_ps(_mm256_load_ps(_ptr + i), _mm256_load_ps(tns._ptr + i)));

        return *this;
    }

    // -= 
    tensor& operator-=(const tensor &tns) {
        for(uint32_t i = 0; i < _size; i += 8) 
            _mm256_store_ps(_ptr + i, _mm256_sub_ps(_mm256_load_ps(_ptr + i), _mm256_load_ps(tns._ptr + i)));

        return *this;
    }

    // *=
    tensor& operator*=(const tensor &tns) {
        for(uint32_t i = 0; i < _size; i += 8) 
            _mm256_store_ps(_ptr + i, _mm256_mul_ps(_mm256_load_ps(_ptr + i), _mm256_load_ps(tns._ptr + i)));

        return *this;
    }
   
    // += value
    tensor& operator+=(const float value) {
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(uint32_t i = 0; i < _size; i += 8)
            _mm256_store_ps(_ptr + i, _mm256_add_ps(_mm256_load_ps(_ptr + i), val));

        return *this;
    }

    // -= value
    tensor& operator-=(const float value) {
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(uint32_t i = 0; i < _size; i += 8) 
            _mm256_store_ps(_ptr + i, _mm256_sub_ps(_mm256_load_ps(_ptr + i), val));

        return *this;
    }

    // *= value
    tensor& operator*=(const float value) {
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(uint32_t i = 0; i < _size; i += 8) 
            _mm256_store_ps(_ptr + i, _mm256_mul_ps(_mm256_load_ps(_ptr + i), val));

        return *this;
    }

    // +
    tensor operator+(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(uint32_t i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw._ptr, _mm256_add_ps(_mm256_load_ps(_ptr + i), _mm256_load_ps(tns._ptr + i)));

        return nw;
    }

    // -
    tensor operator-(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(uint32_t i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw._ptr, _mm256_sub_ps(_mm256_load_ps(_ptr + i), _mm256_load_ps(tns._ptr + i)));
            
        return nw;
    }

    // *
    tensor operator*(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(uint32_t i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw._ptr, _mm256_mul_ps(_mm256_load_ps(_ptr + i), _mm256_load_ps(tns._ptr + i)));

        return nw;
    }
   
    // + value
    tensor operator+(const float value) {
        tensor nw(rows, cols, channels);
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(uint32_t i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw._ptr + i, _mm256_add_ps(_mm256_load_ps(_ptr + i), val));

        return nw;
    }

    // - value
    tensor operator-(const float value) {
        tensor nw(rows, cols, channels);
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(uint32_t i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw._ptr + i, _mm256_sub_ps(_mm256_load_ps(_ptr + i), val));

        return nw;
    }

    // * value
    tensor operator*(const float value) {
        tensor nw(rows, cols, channels);
        __m256 val = _mm256_set_ps(value, value, value, value, value, value, value, value);
        for(uint32_t i = 0; i < _size; i += 8) 
            _mm256_store_ps(nw._ptr + i, _mm256_mul_ps(_mm256_load_ps(_ptr + i), val));

        return nw;
    }

#elif defined(SSE) // ----- SSE OPERATOR OVERLOADING ----- //
    
    // += 
    tensor& operator+=(const tensor &tns) {
        for(uint32_t i = 0; i < _size; i += 4) 
            _mm_store_ps(_ptr + i, _mm_add_ps(_mm_load_ps(_ptr + i), _mm_load_ps(tns._ptr + i)));

        return *this;
    }

    // -= 
    tensor& operator-=(const tensor &tns) {
        for(uint32_t i = 0; i < _size; i += 4) 
            _mm_store_ps(_ptr + i, _mm_sub_ps(_mm_load_ps(_ptr + i), _mm_load_ps(tns._ptr + i)));

        return *this;
    }

    // *=
    tensor& operator*=(const tensor &tns) {
        for(uint32_t i = 0; i < _size; i += 4) 
            _mm_store_ps(_ptr + i, _mm_mul_ps(_mm_load_ps(_ptr + i), _mm_load_ps(tns._ptr + i)));

        return *this;
    }
   
    // += value
    tensor& operator+=(const float value) {
        __m128 val = _mm_set_ps(value, value, value, value);
        for(uint32_t i = 0; i < _size; i += 4)
            _mm_store_ps(_ptr + i, _mm_add_ps(_mm_load_ps(_ptr + i), val));

        return *this;
    }

    // -= value
    tensor& operator-=(const float value) {
        __m128 val = _mm_set_ps(value, value, value, value);
        for(uint32_t i = 0; i < _size; i += 4) 
            _mm_store_ps(_ptr + i, _mm_sub_ps(_mm_load_ps(_ptr + i), val));

        return *this;
    }

    // *= value
    tensor& operator*=(const float value) {
        __m128 val = _mm_set_ps(value, value, value, value);
        for(uint32_t i = 0; i < _size; i += 4) 
            _mm_store_ps(_ptr + i, _mm_mul_ps(_mm_load_ps(_ptr + i), val));

        return *this;
    }

    // +
    tensor operator+(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(uint32_t i = 0; i < _size; i += 4) 
            _mm_store_ps(nw._ptr, _mm_add_ps(_mm_load_ps(_ptr + i), _mm_load_ps(tns._ptr + i)));

        return nw;
    }

    // -
    tensor operator-(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(uint32_t i = 0; i < _size; i += 4) 
            _mm_store_ps(nw._ptr, _mm_sub_ps(_mm_load_ps(_ptr + i), _mm_load_ps(tns._ptr + i)));
            
        return nw;
    }

    // *
    tensor operator*(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(uint32_t i = 0; i < _size; i += 4) 
            _mm_store_ps(nw._ptr, _mm_mul_ps(_mm_load_ps(_ptr + i), _mm_load_ps(tns._ptr + i)));

        return nw;
    }
   
    // + value
    tensor operator+(const float value) {
        tensor nw(rows, cols, channels);
        __m128 val = _mm_set_ps(value, value, value, value);
        for(uint32_t i = 0; i < _size; i += 4) 
            _mm_store_ps(nw._ptr + i, _mm_add_ps(_mm_load_ps(_ptr + i), val));

        return nw;
    }

    // - value
    tensor operator-(const float value) {
        tensor nw(rows, cols, channels);
        __m128 val = _mm_set_ps(value, value, value, value);
        for(uint32_t i = 0; i < _size; i += 4) 
            _mm_store_ps(nw._ptr + i, _mm_sub_ps(_mm_load_ps(_ptr + i), val));

        return nw;
    }

    // * value
    tensor operator*(const float value) {
        tensor nw(rows, cols, channels);
        __m128 val = _mm_set_ps(value, value, value, value);
        for(uint32_t i = 0; i < _size; i += 4) 
            _mm_store_ps(nw._ptr + i, _mm_mul_ps(_mm_load_ps(_ptr + i), val));

        return nw;
    }

#elif defined(NORMAL) // ----- NORMAL OPERATOR OVERLOADING ----- // 

    // += 
    tensor& operator+=(const tensor &tns) {
        for(uint32_t i = 0; i < _size; i++) _ptr[i] += tns._ptr[i];
        return *this;
    }

    // -= 
    tensor& operator-=(const tensor &tns) {
        for(uint32_t i = 0; i < _size; i++) _ptr[i] -= tns._ptr[i];
        return *this;
    }

    // *=
    tensor& operator*=(const tensor &tns) {
        for(uint32_t i = 0; i < _size; i++) _ptr[i] *= tns._ptr[i];
        return *this;
    }
   
    // += value
    tensor& operator+=(const float value) {
        for(uint32_t i = 0; i < _size; i++) _ptr[i] += value;
        return *this;
    }

    // -= value
    tensor& operator-=(const float value) {
        for(uint32_t i = 0; i < _size; i++) _ptr[i] -= value;
        return *this;
    }

    // *= value
    tensor& operator*=(const float value) {
        for(uint32_t i = 0; i < _size; i++) _ptr[i] *= value;
        return *this;
    }

    // +
    tensor operator+(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(uint32_t i = 0; i < _size; i++) nw._ptr[i] = _ptr[i] + tns._ptr[i];
        return nw;
    }

    // -
    tensor operator-(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(uint32_t i = 0; i < _size; i++) nw._ptr[i] = _ptr[i] - tns._ptr[i];
        return nw;
    }

    // *
    tensor operator*(const tensor &tns) {
        tensor nw(rows, cols, channels);
        for(uint32_t i = 0; i < _size; i++) nw._ptr[i] = _ptr[i] * tns._ptr[i];
        return nw;
    }
   
    // + value
    tensor operator+(const float value) {
        tensor nw(rows, cols, channels);
        for(uint32_t i = 0; i < _size; i++) nw._ptr[i] = _ptr[i] + value;
        return nw;
    }

    // - value
    tensor operator-(const float value) {
        tensor nw(rows, cols, channels);
        for(uint32_t i = 0; i < _size; i++) nw._ptr[i] = _ptr[i] - value;
        return nw;
    }

    // * value
    tensor operator*(const float value) {
        tensor nw(rows, cols, channels);
        for(uint32_t i = 0; i < _size; i++) nw._ptr[i] = _ptr[i] * value;
        return nw;
    }

#endif // ----- END OF OPERATOR OVERLOADING ----- //
};

} // namespace pinguml
