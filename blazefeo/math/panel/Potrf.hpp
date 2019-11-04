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
    BLAZE_ALWAYS_INLINE void potrf_backend(size_t k,
        PanelMatrix<MT1, rowMajor> const& A, PanelMatrix<MT2, rowMajor>& L)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;
            
        size_t const M = rows(A);
        size_t const K = k * TILE_SIZE;

        auto L1 = submatrix(~L, K, K, M - K, TILE_SIZE, unchecked);
        gemm_nt(ET(-1.), ET(1.),
            submatrix(~L, K, 0, M - K, K, unchecked),
            submatrix(~L, K, 0, TILE_SIZE, K, unchecked),
            submatrix(~A, K, K, M - K, TILE_SIZE, unchecked),
            L1);

        RegisterMatrix<ET, 1, 1, TILE_SIZE> ker;
        load(ker, tile(L, k, k), spacing(L));
        ker.potrf();
        store(ker, tile(L, k, k), spacing(L));

        for (size_t i = k + 1; (i + 1) * TILE_SIZE <= M; ++i)
            trsm<false, false, true>(ker, tile(L, i, k), tile(L, i, k));
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

        RegisterMatrix<ET, 1, 1, TILE_SIZE> ker;

        size_t k = 0;

        if (TILE_SIZE <= M)
        {
            // Zero-out upper blocks
            // std::fill_n(tile(L, 0, 1), TILE_SIZE * (M - TILE_SIZE), ET {});

            load(ker, tile(A, 0, 0), spacing(A));
            ker.potrf();
            store(ker, tile(L, 0, 0), spacing(L));

            for (size_t i = 1; (i + 1) * TILE_SIZE <= M; ++i)
                trsm<false, false, true>(ker, tile(A, i, 0), tile(L, i, 0));
        }


        if (TILE_SIZE * (++k + 1) <= M)
            potrf_backend(k, A, L); // k = 1

        if (TILE_SIZE * (++k + 1) <= M)
            potrf_backend(k, A, L); // k = 2

        if (TILE_SIZE * (++k + 1) <= M)
            potrf_backend(k, A, L); // k = 3

        if (TILE_SIZE * (++k + 1) <= M)
            potrf_backend(k, A, L); // k = 4

        if (TILE_SIZE * (++k + 1) <= M)
            potrf_backend(k, A, L); // k = 5
        
        while (TILE_SIZE * (++k + 1) <= M) 
            potrf_backend(k, A, L); // k = 6,7,...
    }
}