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


    template <size_t BLOCK_SIZE, typename MT1, typename MT2>
    BLAZE_ALWAYS_INLINE void potrf_backend(
        PanelMatrix<MT1, rowMajor> const& A, PanelMatrix<MT2, rowMajor>& L)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;
        size_t constexpr TILES_PER_BLOCK = BLOCK_SIZE / TILE_SIZE;
            
        size_t const M = rows(A);

        // Unrolling this loop for statically-sized matrices 
        // gives a performance boost up to 10% for matrix sizes below 50.
        // #pragma unroll
        for (size_t k = 0; k * TILE_SIZE + BLOCK_SIZE <= M; k += TILES_PER_BLOCK) 
        {
            size_t const K = k * TILE_SIZE;

            // TODO: improve performance here by using symmetric rank-1 update for the first row of blocks.
            auto L1 = submatrix(~L, K, K, M - K, BLOCK_SIZE, unchecked);
            gemm_nt(ET(-1.), ET(1.),
                submatrix(~L, K, 0, M - K, K, unchecked),
                submatrix(~L, K, 0, BLOCK_SIZE, K, unchecked),
                submatrix(~A, K, K, M - K, BLOCK_SIZE, unchecked),
                L1);

            RegisterMatrix<ET, TILES_PER_BLOCK, BLOCK_SIZE, TILE_SIZE> ker;
            load(ker, tile(L, k, k), spacing(L));
            ker.potrf();
            store(ker, tile(L, k, k), spacing(L));

            // TODO: replace by a call to matrix trsm when it is implemented
            for (size_t i = k + TILES_PER_BLOCK; i * TILE_SIZE + BLOCK_SIZE <= M; i += TILES_PER_BLOCK)
            {
                load(ker, tile(L, i, k), spacing(L));
                trsm<false, false, true>(ker, tile(L, k, k), spacing(L));
                store(ker, tile(L, i, k), spacing(L));
            }

            size_t const rem = M % BLOCK_SIZE;
            if (rem)
            {
                // Process the remainder of the column block
                size_t const i = (M / BLOCK_SIZE) * TILES_PER_BLOCK;
                load(ker, tile(L, i, k), spacing(L));
                trsm<false, false, true>(ker, tile(L, k, k), spacing(L));
                store(ker, tile(L, i, k), spacing(L), rem, BLOCK_SIZE);
            }
        }
    }


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

        potrf_backend<2 * TILE_SIZE>(A, L);
    }
}