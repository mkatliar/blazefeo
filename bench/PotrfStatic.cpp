#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/panel/Potrf.hpp>

#include <bench/Benchmark.hpp>
#include <test/Randomize.hpp>

#include <random>
#include <memory>


#define BENCHMARK_POTRF_STATIC(type, size) \
    BENCHMARK_TEMPLATE(BM_potrf_static, type, size);


namespace blazefeo :: benchmark
{
    template <typename Real, size_t M>
    static void BM_potrf_static(::benchmark::State& state)
    {
        StaticPanelMatrix<Real, M, M> A, L;
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


    // BENCHMARK_POTRF_STATIC(double, 1);
    // BENCHMARK_POTRF_STATIC(double, 2);
    // BENCHMARK_POTRF_STATIC(double, 3);
    BENCHMARK_POTRF_STATIC(double, 4);
    // BENCHMARK_POTRF_STATIC(double, 5);
    // BENCHMARK_POTRF_STATIC(double, 6);
    // BENCHMARK_POTRF_STATIC(double, 7);
    BENCHMARK_POTRF_STATIC(double, 8);
    // BENCHMARK_POTRF_STATIC(double, 9);
    // BENCHMARK_POTRF_STATIC(double, 10);
    // BENCHMARK_POTRF_STATIC(double, 11);
    BENCHMARK_POTRF_STATIC(double, 12);
    // BENCHMARK_POTRF_STATIC(double, 13);
    // BENCHMARK_POTRF_STATIC(double, 14);
    // BENCHMARK_POTRF_STATIC(double, 15);
    BENCHMARK_POTRF_STATIC(double, 16);
    // BENCHMARK_POTRF_STATIC(double, 17);
    // BENCHMARK_POTRF_STATIC(double, 18);
    // BENCHMARK_POTRF_STATIC(double, 19);
    BENCHMARK_POTRF_STATIC(double, 20);
    // BENCHMARK_POTRF_STATIC(double, 21);
    // BENCHMARK_POTRF_STATIC(double, 22);
    // BENCHMARK_POTRF_STATIC(double, 23);
    BENCHMARK_POTRF_STATIC(double, 24);
    // BENCHMARK_POTRF_STATIC(double, 25);
    // BENCHMARK_POTRF_STATIC(double, 26);
    // BENCHMARK_POTRF_STATIC(double, 27);
    BENCHMARK_POTRF_STATIC(double, 28);
    // BENCHMARK_POTRF_STATIC(double, 29);
    // BENCHMARK_POTRF_STATIC(double, 30);
    // BENCHMARK_POTRF_STATIC(double, 31);
    BENCHMARK_POTRF_STATIC(double, 32);
    // BENCHMARK_POTRF_STATIC(double, 33);
    // BENCHMARK_POTRF_STATIC(double, 34);
    // BENCHMARK_POTRF_STATIC(double, 35);
    BENCHMARK_POTRF_STATIC(double, 36);
    // BENCHMARK_POTRF_STATIC(double, 37);
    // BENCHMARK_POTRF_STATIC(double, 38);
    // BENCHMARK_POTRF_STATIC(double, 39);
    BENCHMARK_POTRF_STATIC(double, 40);
    // BENCHMARK_POTRF_STATIC(double, 41);
    // BENCHMARK_POTRF_STATIC(double, 42);
    // BENCHMARK_POTRF_STATIC(double, 43);
    BENCHMARK_POTRF_STATIC(double, 44);
    // BENCHMARK_POTRF_STATIC(double, 45);
    // BENCHMARK_POTRF_STATIC(double, 46);
    // BENCHMARK_POTRF_STATIC(double, 47);
    BENCHMARK_POTRF_STATIC(double, 48);
    // BENCHMARK_POTRF_STATIC(double, 49);
    // BENCHMARK_POTRF_STATIC(double, 50);
}
