#include <blazefeo/math/StaticPanelMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename Kernel>
    static void BM_RegisterMatrix_store(State& state)
    {
        using Traits = RegisterMatrixTraits<Kernel>;
        size_t constexpr M = Traits::rows;
        size_t constexpr N = Traits::columns;

        StaticPanelMatrix<double, M, N, rowMajor> c, d;
        randomize(c);

        Kernel ker;
        load(ker, c.tile(0, 0), c.spacing());

        for (auto _ : state)
        {
            store(ker, d.tile(0, 0), d.spacing());
            DoNotOptimize(d);
        }

        state.counters["flops"] = Counter(M * N, Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store, RegisterMatrix<double, 1, 4, 4>);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store, RegisterMatrix<double, 2, 4, 4>);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store, RegisterMatrix<double, 3, 4, 4>);
}