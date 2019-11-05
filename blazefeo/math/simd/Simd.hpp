#pragma once

#include <blaze/util/Types.h>
#include <blaze/system/Inline.h>

#include <immintrin.h>


namespace blazefeo
{
    using namespace blaze;


    template <typename T, size_t SIMD_SIZE>
    struct Simd;


    template <>
    struct Simd<double, 4>
    {
        using IntrinsicType = __m256d;
    };


    template <size_t SS, typename T>
    auto load(T const * ptr);


    template <>
    inline auto load<4, double>(double const * ptr)
    {
        return _mm256_load_pd(ptr);
    }


    inline void store(double * ptr, __m256d a)
    {
        _mm256_store_pd(ptr, a);
    }
}