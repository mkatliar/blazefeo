#pragma once

#include <blazefeo/math/dense/GemmBackend.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/math/DenseMatrix.h>

#include <algorithm>


namespace blazefeo
{
    template <typename ST1, typename ST2, typename MT1, typename MT2, typename MT3, typename MT4>
    BLAZE_ALWAYS_INLINE void gemm(
        ST1 alpha,
        DenseMatrix<MT1, columnMajor> const& A, DenseMatrix<MT2, rowMajor> const& B, 
        ST2 beta, DenseMatrix<MT3, columnMajor> const& C, DenseMatrix<MT4, columnMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT4>, ET);

        size_t const M = rows(A);
        size_t const N = columns(B);
        size_t const K = columns(A);

        if (rows(B) != K)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(C) != M || columns(C) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(D) != M || columns(D) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        size_t i = 0;

        // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
        // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
        for (; i + 2 * TILE_SIZE < M && i + 4 * TILE_SIZE != M; i += 3 * TILE_SIZE)
            gemm_backend<3 * TILE_SIZE, TILE_SIZE>(i, alpha, ~A, ~B, beta, ~C, ~D);

        for (; i + 1 * TILE_SIZE < M; i += 2 * TILE_SIZE)
            gemm_backend<2 * TILE_SIZE, TILE_SIZE>(i, alpha, ~A, ~B, beta, ~C, ~D);

        for (; i + 0 * TILE_SIZE < M; i += 1 * TILE_SIZE)
            gemm_backend<1 * TILE_SIZE, TILE_SIZE>(i, alpha, ~A, ~B, beta, ~C, ~D);
    }
}