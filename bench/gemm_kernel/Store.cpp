#include <blazefeo/math/StaticPanelMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename Kernel>
    static void BM_GemmKernel_store(State& state)
    {
        using Traits = GemmKernelTraits<Kernel>;
        size_t constexpr M = Traits::rows;
        size_t constexpr N = Traits::columns;

        StaticPanelMatrix<double, M, N, rowMajor> c, d;
        randomize(c);

        Kernel ker(c.tile(0, 0), c.spacing());

        for (auto _ : state)
        {
            ker.store(d.tile(0, 0), d.spacing());
            DoNotOptimize(d);
        }

        state.counters["flops"] = Counter(M * N, Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_GemmKernel_store, GemmKernel<double, 1, 1, 4>);
    BENCHMARK_TEMPLATE(BM_GemmKernel_store, GemmKernel<double, 2, 1, 4>);
    BENCHMARK_TEMPLATE(BM_GemmKernel_store, GemmKernel<double, 3, 1, 4>);
}