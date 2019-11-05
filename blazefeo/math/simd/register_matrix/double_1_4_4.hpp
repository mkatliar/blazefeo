#pragma once

#include <blazefeo/math/simd/register_matrix/RegisterMatrix.hpp>
#include <blazefeo/Exception.hpp>

#include <cmath>


namespace blazefeo
{
    template <>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<double, 1, 4, 4>::load(double beta, double const * ptr, size_t spacing, size_t m, size_t n)
    {
        if (n > 0)
            v_[0][0] = beta * _mm256_load_pd(ptr);

        if (n > 1)
            v_[0][1] = beta * _mm256_load_pd(ptr + 4);

        if (n > 2)
            v_[0][2] = beta * _mm256_load_pd(ptr + 8);

        if (n > 3)
            v_[0][3] = beta * _mm256_load_pd(ptr + 12);
    }


    template <>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<double, 1, 4, 4>::store(double * ptr, size_t spacing, size_t m, size_t n) const
    {
        if (m >= 4)
        {
            if (n > 0)
                _mm256_store_pd(ptr, v_[0][0]);

            if (n > 1)
                _mm256_store_pd(ptr + 4, v_[0][1]);

            if (n > 2)
                _mm256_store_pd(ptr + 8, v_[0][2]);

            if (n > 3)
                _mm256_store_pd(ptr + 12, v_[0][3]);
        }
        else if (m > 0)
        {
            __m256i const mask = _mm256_set_epi64x(
                m > 3 ? 0x8000000000000000ULL : 0, 
                m > 2 ? 0x8000000000000000ULL : 0,
                m > 1 ? 0x8000000000000000ULL : 0,
                m > 0 ? 0x8000000000000000ULL : 0); 

            if (n > 0)
                _mm256_maskstore_pd(ptr, mask, v_[0][0]);

            if (n > 1)
                _mm256_maskstore_pd(ptr + 4, mask, v_[0][1]);

            if (n > 2)
                _mm256_maskstore_pd(ptr + 8, mask, v_[0][2]);

            if (n > 3)
                _mm256_maskstore_pd(ptr + 12, mask, v_[0][3]);
        }
    }


    template <>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<double, 1, 4, 4>::potrf()
    {
        v_[0][0] *= 1. / std::sqrt(v_[0][0][0]);
        
        v_[0][1] = _mm256_fnmadd_pd(_mm256_set_pd(v_[0][0][1], v_[0][0][1], v_[0][0][1], v_[0][0][1]), v_[0][0], v_[0][1]);
        v_[0][1] *= 1. / std::sqrt(v_[0][1][1]);

        v_[0][2] = _mm256_fnmadd_pd(_mm256_set_pd(v_[0][0][2], v_[0][0][2], v_[0][0][2], v_[0][0][2]), v_[0][0], v_[0][2]);
        v_[0][2] = _mm256_fnmadd_pd(_mm256_set_pd(v_[0][1][2], v_[0][1][2], v_[0][1][2], v_[0][1][2]), v_[0][1], v_[0][2]);
        v_[0][2] *= 1. / std::sqrt(v_[0][2][2]);

        v_[0][3] = _mm256_fnmadd_pd(_mm256_set_pd(v_[0][0][3], v_[0][0][3], v_[0][0][3], v_[0][0][3]), v_[0][0], v_[0][3]);
        v_[0][3] = _mm256_fnmadd_pd(_mm256_set_pd(v_[0][1][3], v_[0][1][3], v_[0][1][3], v_[0][1][3]), v_[0][1], v_[0][3]);
        v_[0][3] = _mm256_fnmadd_pd(_mm256_set_pd(v_[0][2][3], v_[0][2][3], v_[0][2][3], v_[0][2][3]), v_[0][2], v_[0][3]);
        v_[0][3] *= 1. / std::sqrt(v_[0][3][3]);
    }


    template <>
    template <>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<double, 1, 4, 4>::ger<false, true>(double alpha, double const * a, size_t sa, double const * b, size_t sb)
    {
        __m256d const a_v0 = alpha * _mm256_load_pd(a);
        v_[0][0] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 0), v_[0][0]);
        v_[0][1] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 1), v_[0][1]);
        v_[0][2] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 2), v_[0][2]);
        v_[0][3] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 3), v_[0][3]);
    }


    template <>
    template <>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<double, 1, 4, 4>::ger<false, true>(double alpha, double const * a, size_t sa, double const * b, size_t sb, size_t m, size_t n)
    {
        __m256d const a_v0 = alpha * _mm256_load_pd(a);

        if (n > 0)
            v_[0][0] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b), v_[0][0]);

        if (n > 1)
            v_[0][1] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 1), v_[0][1]);

        if (n > 2)
            v_[0][2] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 2), v_[0][2]);

        if (n > 3)
            v_[0][3] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 3), v_[0][3]);
    }


    template <>
    template <>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<double, 1, 4, 4>::trsm<false, false, true>(double const * a, double * x) const
    {
        __m256d xx[4];
        xx[0] = _mm256_load_pd(a + 0);
        xx[0] /= v_[0][0][0];
        _mm256_store_pd(x + 0, xx[0]);

        xx[1] = _mm256_load_pd(a + 4);
        xx[1] = _mm256_fnmadd_pd(_mm256_set_pd(v_[0][0][1], v_[0][0][1], v_[0][0][1], v_[0][0][1]), xx[0], xx[1]);
        xx[1] /= v_[0][1][1];
        _mm256_store_pd(x + 4, xx[1]);

        xx[2] = _mm256_load_pd(a + 8);
        xx[2] = _mm256_fnmadd_pd(_mm256_set_pd(v_[0][0][2], v_[0][0][2], v_[0][0][2], v_[0][0][2]), xx[0], xx[2]);
        xx[2] = _mm256_fnmadd_pd(_mm256_set_pd(v_[0][1][2], v_[0][1][2], v_[0][1][2], v_[0][1][2]), xx[1], xx[2]);
        xx[2] /= v_[0][2][2];
        _mm256_store_pd(x + 8, xx[2]);

        xx[3] = _mm256_load_pd(a + 12);
        xx[3] = _mm256_fnmadd_pd(_mm256_set_pd(v_[0][0][3], v_[0][0][3], v_[0][0][3], v_[0][0][3]), xx[0], xx[3]);
        xx[3] = _mm256_fnmadd_pd(_mm256_set_pd(v_[0][1][3], v_[0][1][3], v_[0][1][3], v_[0][1][3]), xx[1], xx[3]);
        xx[3] = _mm256_fnmadd_pd(_mm256_set_pd(v_[0][2][3], v_[0][2][3], v_[0][2][3], v_[0][2][3]), xx[2], xx[3]);
        xx[3] /= v_[0][3][3];
        _mm256_store_pd(x + 12, xx[3]);
    }
}