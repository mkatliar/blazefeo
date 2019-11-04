#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/panel/register_matrix/double_1_1_4.hpp>
#include <blazefeo/math/panel/register_matrix/double_2_1_4.hpp>
#include <blazefeo/math/panel/register_matrix/double_3_1_4.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename T, size_t M, size_t N, size_t P>
    static void BM_RegisterMatrix_potrf(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, P>;
        using Traits = RegisterMatrixTraits<Kernel>;
        size_t constexpr m = Traits::rows;
        size_t constexpr n = Traits::columns;
        
        StaticPanelMatrix<double, m, n, rowMajor> a;
        randomize(a);

        Kernel ker;
        load(ker, a.tile(0, 0), a.spacing());

        for (auto _ : state)
        {
            ker.potrf();
            DoNotOptimize(ker);
        }

        if (m == n)
            state.counters["flops"] = Counter(m * m * m / 3., Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, double, 1, 1, 4);
    // BENCHMARK_TEMPLATE(BM_RegisterMatrix_gemm_nt, double, 2, 1, 4);
    // BENCHMARK_TEMPLATE(BM_RegisterMatrix_gemm_nt, double, 3, 1, 4);
}