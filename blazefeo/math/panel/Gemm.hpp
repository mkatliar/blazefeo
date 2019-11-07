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


    template <size_t KM, size_t KN, typename ST1, typename ST2, typename MT1, typename MT2, typename MT3, typename MT4>
    void gemm_nt_backend(
        ST1 alpha, ST2 beta,
        PanelMatrix<MT1, rowMajor> const& A, PanelMatrix<MT2, rowMajor> const& B, 
        PanelMatrix<MT3, rowMajor> const& C, PanelMatrix<MT4, rowMajor>& D);


    template <typename ST1, typename ST2, typename MT1, typename MT2, typename MT3, typename MT4>
    inline void gemm_nt(
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

        // The algoritm of kernel selection is defined below and probably can be improved.
        size_t constexpr KN = 4;

        if (M == 16)
        {
            // This branch is to improve performance in case when the remaining number of rows is 16:
            // it is more efficient to apply an 8x-kernel 2 times than a 12x-kernel + a 4x-kernel.            
            gemm_nt_backend<8, KN>(alpha, beta, ~A, ~B, ~C, ~D);
        }
        else if (M >= 12)
            gemm_nt_backend<12, KN>(alpha, beta, ~A, ~B, ~C, ~D);
        else if (M >= 8)
            gemm_nt_backend<8, KN>(alpha, beta, ~A, ~B, ~C, ~D);
        else
        {
            // The smallest kernel that will take care of everything else
            gemm_nt_backend<4, KN>(alpha, beta, ~A, ~B, ~C, ~D);
        }
    }


    template <size_t KM, size_t KN, typename ST1, typename ST2, typename MT1, typename MT2, typename MT3, typename MT4>
    BLAZE_ALWAYS_INLINE void gemm_nt_backend(
        ST1 alpha, ST2 beta,
        PanelMatrix<MT1, rowMajor> const& A, PanelMatrix<MT2, rowMajor> const& B, 
        PanelMatrix<MT3, rowMajor> const& C, PanelMatrix<MT4, rowMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_STATIC_ASSERT(KM % TILE_SIZE == 0);

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT4>, ET);

        size_t const M = rows(A);
        size_t const N = rows(B);
        size_t const K = columns(A);

        BLAZE_USER_ASSERT(columns(B) == K, "Matrix sizes do not match");
        BLAZE_USER_ASSERT(rows(C) == M && columns(C) == N, "Matrix sizes do not match");
        BLAZE_USER_ASSERT(rows(D) == M && columns(D) == N, "Matrix sizes do not match");

        if (M < TILE_SIZE)
        {
            // Use a small kernel with partial save to calculate the bottom of the resulting matrix.
            RegisterMatrix<ET, 1, TILE_SIZE, TILE_SIZE> ker;

            size_t const rm = M;
            size_t j = 0;
            ET const * b = tile(B, 0, 0);

            for (; j + TILE_SIZE <= N; j += TILE_SIZE)
                gemm_backend<false, true>(ker, K, alpha, beta,
                    ptr(A, 0, 0), spacing(A), ptr(B, j, 0), spacing(B),
                    ptr(C, 0, j), spacing(C), ptr(D, 0, j), spacing(D), rm, TILE_SIZE);

            if (j < N)
                gemm_backend<false, true>(ker, K, alpha, beta,
                    ptr(A, 0, 0), spacing(A), ptr(B, j, 0), spacing(B),
                    ptr(C, 0, j), spacing(C), ptr(D, 0, j), spacing(D), rm, N - j);
        }
        else
        {
            size_t i = 0;
            ET const * a = tile(A, 0, 0);

            for (; i + KM <= M; i += KM, a += KM / TILE_SIZE * spacing(A))
            {
                RegisterMatrix<ET, KM / TILE_SIZE, KN, TILE_SIZE> ker;
                size_t j = 0;

                for (; j + KN <= N; j += KN)
                    gemm_backend<false, true>(ker, K, alpha, beta,
                        a, spacing(A), ptr(B, j, 0), spacing(B),
                        ptr(C, i, j), spacing(C), ptr(D, i, j), spacing(D));

                if (j < N)
                    gemm_backend<false, true>(ker, K, alpha, beta,
                        a, spacing(A), ptr(B, j, 0), spacing(B),
                        ptr(C, i, j), spacing(C), ptr(D, i, j), spacing(D), KM, N - j);
            }

            // Calculate the rest with a smaller kernel.
            if (i < M)
            {
                auto D1 = submatrix(~D, i, 0, M - i, N);
                gemm_nt(alpha, beta, submatrix(~A, i, 0, M - i, K), ~B, submatrix(~C, i, 0, M - i, N), ~D1);
            }
        }
    }
}