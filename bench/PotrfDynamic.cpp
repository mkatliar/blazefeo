#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/panel/Potrf.hpp>

#include <bench/Benchmark.hpp>
#include <test/Randomize.hpp>

#include <random>
#include <memory>


namespace blazefeo :: benchmark
{
    template <typename Real>
    static void BM_potrf_dynamic(::benchmark::State& state)
    {
        size_t const M = state.range(0);

        DynamicPanelMatrix<Real, rowMajor> A(M, M), L(M, M);
        makePositiveDefinite(A);

        for (auto _ : state)
        {
            potrf(A, L);
            DoNotOptimize(A);
            DoNotOptimize(L);
        }

        state.counters["flops"] = Counter(M * M * M / 3., Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
    }


    BENCHMARK_TEMPLATE(BM_potrf_dynamic, double)->DenseRange(4, 300, 4);
}
