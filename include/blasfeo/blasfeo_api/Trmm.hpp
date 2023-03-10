// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blasfeo_d_aux.h>
#include <blasfeo_s_aux.h>
#include <blasfeo_d_blasfeo_api.h>
#include <blasfeo_s_blasfeo_api.h>

#include <blasfeo/SizeT.hpp>


namespace blasfeo
{
    /// @brief D <= alpha * B * A ; A lower triangular
    ///
    /// m, n: "Rows and columns of B according to the netlib docs" according to @zanellia
    inline void trmm_rlnn(size_t m, size_t n, double alpha,
        blasfeo_dmat const& sA, size_t ai, size_t aj,
        blasfeo_dmat const& sB, size_t bi, size_t bj,
        blasfeo_dmat& sD, size_t di, size_t dj)
    {
        ::blasfeo_dtrmm_rlnn(m, n, alpha, const_cast<blasfeo_dmat *>(&sA), ai, aj, const_cast<blasfeo_dmat *>(&sB), bi, bj, &sD, di, dj);
    }


    /// @brief D <= alpha * B * A ; A lower triangular
    ///
    /// m, n: "Rows and columns of B according to the netlib docs" according to @zanellia
    inline void trmm_rlnn(size_t m, size_t n, float alpha,
        blasfeo_smat const& sA, size_t ai, size_t aj,
        blasfeo_smat const& sB, size_t bi, size_t bj,
        blasfeo_smat& sD, size_t di, size_t dj)
    {
        ::blasfeo_strmm_rlnn(m, n, alpha, const_cast<blasfeo_smat *>(&sA), ai, aj, const_cast<blasfeo_smat *>(&sB), bi, bj, &sD, di, dj);
    }


    /// @brief D <= alpha * B * A^T ; B upper triangular
    inline void trmm_rutn(size_t m, size_t n, double alpha,
        blasfeo_dmat& sA, size_t ai, size_t aj,
        blasfeo_dmat& sB, size_t bi, size_t bj,
        blasfeo_dmat& sD, size_t di, size_t dj)
    {
        ::blasfeo_dtrmm_rutn(m, n, alpha, const_cast<blasfeo_dmat *>(&sA), ai, aj, const_cast<blasfeo_dmat *>(&sB), bi, bj, &sD, di, dj);
    }


    /// @brief D <= alpha * B * A^T ; B upper triangular
    inline void trmm_rutn(size_t m, size_t n, float alpha,
        blasfeo_smat& sA, size_t ai, size_t aj,
        blasfeo_smat& sB, size_t bi, size_t bj,
        blasfeo_smat& sD, size_t di, size_t dj)
    {
        ::blasfeo_strmm_rutn(m, n, alpha, const_cast<blasfeo_smat *>(&sA), ai, aj, const_cast<blasfeo_smat *>(&sB), bi, bj, &sD, di, dj);
    }
}
