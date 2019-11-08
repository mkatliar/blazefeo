#pragma once

#include <blazefeo/math/PanelMatrix.hpp>
#include <blazefeo/math/views/submatrix/Panel.hpp>
#include <blazefeo/math/panel/Gemm.hpp>
#include <blazefeo/system/Tile.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>

#include <algorithm>


namespace blazefeo
{
    using namespace blaze;


    template <size_t KM, size_t KN, typename MT1, typename MT2>
    BLAZE_ALWAYS_INLINE void potrf_backend(size_t k, size_t i,
        PanelMatrix<MT1, rowMajor> const& A, PanelMatrix<MT2, rowMajor>& L)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;
        
        size_t const M = rows(A);
        size_t const N = columns(A);

        RegisterMatrix<ET, KM / TILE_SIZE, KN, TILE_SIZE> ker;

        load(ker, ptr(A, i, k), spacing(A));

        for (size_t l = 0; l < k; ++l)
            ger<false, true>(ker, ET(-1.), ptr(L, i, l), spacing(L), ptr(L, k, l), spacing(L));

        if (i == k)
            ker.potrf();
        else
            trsm<false, false, true>(ker, ptr(L, k, k), spacing(L));

        store(ker, ptr(L, i, k), spacing(L));
    }


    template <typename MT1, typename MT2>
    BLAZE_ALWAYS_INLINE void potrf(
        PanelMatrix<MT1, rowMajor> const& A, PanelMatrix<MT2, rowMajor>& L)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);

        size_t const M = rows(A);
        size_t const N = columns(A);

        if (columns(A) > M)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid matrix size");

        if (rows(L) != M)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid matrix size");

        if (columns(L) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid matrix size");

        size_t constexpr KN = TILE_SIZE;
        size_t k = 0;

        for (; k < N; k += KN)
        {
            size_t i = k;

            // for (; i + 2 * TILE_SIZE < M; i += 3 * TILE_SIZE)
            //     potrf_backend<3 * TILE_SIZE, KN>(k, i, ~A, ~L);

            // for (; i + 1 * TILE_SIZE < M; i += 2 * TILE_SIZE)
            //     potrf_backend<2 * TILE_SIZE, KN>(k, i, ~A, ~L);

            for (; i + 0 * TILE_SIZE < M; i += 1 * TILE_SIZE)
                potrf_backend<1 * TILE_SIZE, KN>(k, i, ~A, ~L);
        }
    }
}