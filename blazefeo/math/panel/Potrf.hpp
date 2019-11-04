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


    template <typename MT1, typename MT2>
    BLAZE_ALWAYS_INLINE void potrf(
        PanelMatrix<MT1, rowMajor> const& A, PanelMatrix<MT2, rowMajor>& L)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);

        size_t const M = rows(A);

        if (columns(A) != M)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid matrix size");

        if (rows(L) != M)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid matrix size");

        if (columns(L) != M)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid matrix size");

        GemmKernel<ET, 1, 1, TILE_SIZE> ker;

        for (size_t k = 0; (k + 1) * TILE_SIZE <= M; ++k)
        {
            size_t i = 0;

            // Zero-out upper blocks
            for (; i < k; ++i)
                std::fill_n(tile(L, i, k), TILE_SIZE * TILE_SIZE, ET {});

            size_t const K = k * TILE_SIZE;
            auto L1 = submatrix(~L, K, K, M - K, TILE_SIZE);
            gemm_nt(ET(-1.), ET(1.),
                submatrix(~L, K, 0, M - K, K),
                submatrix(~L, K, 0, TILE_SIZE, K),
                submatrix(~A, K, K, M - K, TILE_SIZE),
                L1);

            // for (; (i + 1) * TILE_SIZE <= M; ++i)
            // {   
            //     gemm_backend<false, true>(ker, K, -1., 1.,
            //         tile(L, i, 0), spacing(L), tile(L, k, 0), spacing(L),
            //         tile(A, i, k), spacing(A), tile(L, i, k), spacing(L));
            // }

            load(ker, tile(L, k, k), spacing(L));
            ker.potrf();
            store(ker, tile(L, k, k), spacing(L));

            for (size_t i = k + 1; (i + 1) * TILE_SIZE <= M; ++i)
                trsm<false, false, true>(ker, tile(L, i, k), tile(L, i, k));
        }
    }
}