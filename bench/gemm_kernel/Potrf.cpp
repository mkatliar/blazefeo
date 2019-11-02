#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/panel/gemm/GemmKernel_double_1_1_4.hpp>
#include <blazefeo/math/panel/gemm/GemmKernel_double_2_1_4.hpp>
#include <blazefeo/math/panel/gemm/GemmKernel_double_3_1_4.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename Kernel>
    static void BM_GemmKernel_potrf(State& state)
    {
        using Traits = GemmKernelTraits<Kernel>;
        size_t constexpr M = Traits::rows;
        size_t constexpr N = Traits::columns;
        
        StaticPanelMatrix<double, M, N, rowMajor> a;
        randomize(a);

        Kernel ker(a.tile(0, 0), a.spacing());

        for (auto _ : state)
        {
            ker.potrf();
            DoNotOptimize(ker);
        }

        // state.counters["flops"] = Counter(M * N * K, Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_GemmKernel_potrf, GemmKernel<double, 1, 1, 4>);
    // BENCHMARK_TEMPLATE(BM_GemmKernel_gemm_nt, GemmKernel<double, 2, 1, 4>);
    // BENCHMARK_TEMPLATE(BM_GemmKernel_gemm_nt, GemmKernel<double, 3, 1, 4>);
}