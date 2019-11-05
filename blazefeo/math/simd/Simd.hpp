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
}