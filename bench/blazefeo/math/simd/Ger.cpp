// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename T, size_t M, size_t N, bool SO>
    static void BM_RegisterMatrix_ger_nt(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        using Traits = RegisterMatrixTraits<Kernel>;
        using ET = ElementType_t<Kernel>;
        size_t constexpr K = 100;
        
        DynamicPanelMatrix<ET> a(Traits::rows, K);
        DynamicPanelMatrix<ET> b(Traits::columns, K);
        DynamicPanelMatrix<ET> c(Traits::rows, Traits::columns);

        randomize(a);
        randomize(b);
        randomize(c);

        Kernel ker;
        load(ker, c.ptr(0, 0), c.spacing());

        for (auto _ : state)
        {
            for (size_t i = 0; i < K; ++i)
                ger<a.storageOrder, !decltype(b)::storageOrder>(ker, ET(0.1), ptr(a, 0, i), spacing(a), ptr(b, 0, i), spacing(b));

            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(2 * M * N * K, Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, double, 4, 8, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, double, 12, 4, columnMajor);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 16, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 24, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 16, 5, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 16, 6, columnMajor);
}