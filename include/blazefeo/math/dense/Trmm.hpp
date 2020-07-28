#pragma once

#include <blazefeo/math/dense/GemmBackend.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/math/DenseMatrix.h>

#include <algorithm>


namespace blazefeo
{
    /// @brief C = alpha * A * B + C; A lower-triangular
    ///
    template <typename ST, typename MT1, typename MT2, typename MT3>
    inline void trmm(
        ST alpha,
        DenseMatrix<MT1, columnMajor> const& A, DenseMatrix<MT2, rowMajor> const& B, 
        DenseMatrix<MT3, columnMajor>& C)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);

        size_t const M = rows(A);
        size_t const N = columns(B);

        if (columns(A) != M)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(B) != M)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        size_t i = 0;

        // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
        // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
        for (; i + 2 * TILE_SIZE < M && i + 4 * TILE_SIZE != M; i += 3 * TILE_SIZE)
            gemm_backend<3 * TILE_SIZE, TILE_SIZE>(i, alpha, ~A, ~B, 0., ~C, ~C);

        for (; i + 1 * TILE_SIZE < M; i += 2 * TILE_SIZE)
            gemm_backend<2 * TILE_SIZE, TILE_SIZE>(i, alpha, ~A, ~B, 0., ~C, ~C);

        for (; i + 0 * TILE_SIZE < M; i += 1 * TILE_SIZE)
            gemm_backend<1 * TILE_SIZE, TILE_SIZE>(i, alpha, ~A, ~B, 0., ~C, ~C);
    }
}