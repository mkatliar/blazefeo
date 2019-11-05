#pragma once

#include <blazefeo/math/PanelMatrix.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>
#include <blazefeo/system/Tile.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>

#include <algorithm>


namespace blazefeo
{
    using namespace blaze;


    template <bool TA, bool TB, typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void gemm_backend(RegisterMatrix<T, M, N, BS>& ker, size_t K, T alpha, T beta,
        T const * a, size_t sa, T const * b, size_t sb, T const * c, size_t sc, T * d, size_t sd)
    {
        load(ker, beta, c, sc);

        for (size_t k = 0; k < K; ++k)
        {
            ger<TA, TB>(ker, alpha, a, sa, b, sb);

            a += TA ? M * sa : BS;
            b += TB ? BS : N * sb;
        }

        store(ker, d, sd);
    }


    template <bool TA, bool TB, typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void gemm_backend(RegisterMatrix<T, M, N, BS>& ker, size_t K, T alpha, T beta,
        T const * a, size_t sa, T const * b, size_t sb, T const * c, size_t sc, T * d, size_t sd,
        size_t md, size_t nd)
    {
        load(ker, beta, c, sc, md, nd);
        
        for (size_t k = 0; k < K; ++k)
        {
            ger<TA, TB>(ker, alpha, a, sa, b, sb, md, nd);

            a += TA ? M * sa : BS;
            b += TB ? BS : N * sb;
        }

        store(ker, d, sd, md, nd);
    }


    template <typename MT1, typename MT2, typename MT3, typename MT4>
    BLAZE_ALWAYS_INLINE void gemm_nt(
        PanelMatrix<MT1, rowMajor> const& A, PanelMatrix<MT2, rowMajor> const& B, 
        PanelMatrix<MT3, rowMajor> const& C, PanelMatrix<MT4, rowMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT4>, ET);

        gemm_nt(ET(1.), ET(1.), ~A, ~B, ~C, ~D);
    }


    template <typename ST1, typename ST2, typename MT1, typename MT2, typename MT3, typename MT4>
    BLAZE_ALWAYS_INLINE void gemm_nt(
        ST1 alpha, ST2 beta,
        PanelMatrix<MT1, rowMajor> const& A, PanelMatrix<MT2, rowMajor> const& B, 
        PanelMatrix<MT3, rowMajor> const& C, PanelMatrix<MT4, rowMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT4>, ET);

        size_t const M = rows(A);
        size_t const N = rows(B);
        size_t const K = columns(A);

        if (columns(B) != K)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(C) != M || columns(C) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(D) != M || columns(D) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        size_t i = 0;
        ET const * a = tile(A, 0, 0);

        // (i + 4) * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
        // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
        for (; (i + 3) * TILE_SIZE <= M && (i + 4) * TILE_SIZE != M; i += 3, a += 3 * spacing(A))
        {
            RegisterMatrix<ET, 3, TILE_SIZE, TILE_SIZE> ker;
            size_t j = 0;

            for (; (j + 1) * TILE_SIZE <= N; ++j)
                gemm_backend<false, true>(ker, K, alpha, beta,
                    a, spacing(A), tile(B, j, 0), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D));

            for (; j * TILE_SIZE < N; ++j)
                gemm_backend<false, true>(ker, K, alpha, beta,
                    a, spacing(A), tile(B, j, 0), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D), 3 * TILE_SIZE, std::min(N - j * TILE_SIZE, TILE_SIZE));
        }

        for (; (i + 2) * TILE_SIZE <= M; i += 2, a += 2 * spacing(A))
        {
            RegisterMatrix<ET, 2, TILE_SIZE, TILE_SIZE> ker;
            size_t j = 0;

            for (; (j + 1) * TILE_SIZE <= N; ++j)
                gemm_backend<false, true>(ker, K, alpha, beta,
                    a, spacing(A), tile(B, j, 0), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D));

            for (; j * TILE_SIZE < N; ++j)
                gemm_backend<false, true>(ker, K, alpha, beta,
                    a, spacing(A), tile(B, j, 0), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D), 2 * TILE_SIZE, std::min(N - j * TILE_SIZE, TILE_SIZE));
        }

        for (; (i + 1) * TILE_SIZE <= M; ++i, a += spacing(A))
        {
            RegisterMatrix<ET, 1, TILE_SIZE, TILE_SIZE> ker;

            size_t j = 0;
            ET const * b = tile(B, 0, 0);

            for (; (j + 1) * TILE_SIZE <= N; ++j)
                gemm_backend<false, true>(ker, K, alpha, beta,
                    a, spacing(A), b + j * spacing(B), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D));

            for (; j * TILE_SIZE < N; ++j)
                gemm_backend<false, true>(ker, K, alpha, beta,
                    a, spacing(A), b + j * spacing(B), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D), TILE_SIZE, std::min(N - j * TILE_SIZE, TILE_SIZE));
        }

        for (; i * TILE_SIZE < M; ++i, a += spacing(A))
        {
            RegisterMatrix<ET, 1, TILE_SIZE, TILE_SIZE> ker;

            size_t const rm = std::min(M - i * TILE_SIZE, TILE_SIZE);
            size_t j = 0;
            ET const * b = tile(B, 0, 0);

            for (; (j + 1) * TILE_SIZE <= N; ++j)
                gemm_backend<false, true>(ker, K, alpha, beta,
                    a, spacing(A), b + j * spacing(B), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D), rm, TILE_SIZE);

            for (; j * TILE_SIZE < N; ++j)
                gemm_backend<false, true>(ker, K, alpha, beta,
                    a, spacing(A), b + j * spacing(B), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D), rm, std::min(N - j * TILE_SIZE, TILE_SIZE));
        }
    }
}