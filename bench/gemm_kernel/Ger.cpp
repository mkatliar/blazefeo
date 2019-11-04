#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/panel/gemm/GemmKernel_double_1_1_4.hpp>
#include <blazefeo/math/panel/gemm/GemmKernel_double_2_1_4.hpp>
#include <blazefeo/math/panel/gemm/GemmKernel_double_3_1_4.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename T, size_t M, size_t N, size_t P>
    static void BM_GemmKernel_ger_nt(State& state)
    {
        using Kernel = GemmKernel<T, M, N, P>;
        using Traits = GemmKernelTraits<Kernel>;
        size_t constexpr K = 10000;
        
        DynamicMatrix<double, columnMajor> a(Traits::rows, K);
        DynamicMatrix<double, columnMajor> b(Traits::columns, K);
        StaticPanelMatrix<double, Traits::rows, Traits::columns, rowMajor> c;

        randomize(a);
        randomize(b);
        randomize(c);

        Kernel ker;
        load(ker, c.tile(0, 0), c.spacing());

        for (auto _ : state)
        {
            for (size_t i = 0; i < K; ++i)
                ger<false, true>(ker, 0.1, data(a) + Traits::rows * i, spacing(a), data(b) + Traits::columns * i, spacing(b));

            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(2 * M * N * K, Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_GemmKernel_ger_nt, double, 1, 1, 4);
    // BENCHMARK_TEMPLATE(BM_GemmKernel_ger_nt, double, 2, 1, 4);
    // BENCHMARK_TEMPLATE(BM_GemmKernel_ger_nt, double, 3, 1, 4);
}