#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/panel/Potrf.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <test/Randomize.hpp>

#include <random>
#include <memory>


#define BENCHMARK_POTRF_STATIC(type, size) \
    BENCHMARK_TEMPLATE(BM_potrf_static, type, size);

#define BENCHMARK_POTRF_STATIC_10(type, size) \
    BENCHMARK_TEMPLATE(BM_potrf_static, type, size + 0); \
    BENCHMARK_TEMPLATE(BM_potrf_static, type, size + 1); \
    BENCHMARK_TEMPLATE(BM_potrf_static, type, size + 2); \
    BENCHMARK_TEMPLATE(BM_potrf_static, type, size + 3); \
    BENCHMARK_TEMPLATE(BM_potrf_static, type, size + 4); \
    BENCHMARK_TEMPLATE(BM_potrf_static, type, size + 5); \
    BENCHMARK_TEMPLATE(BM_potrf_static, type, size + 6); \
    BENCHMARK_TEMPLATE(BM_potrf_static, type, size + 7); \
    BENCHMARK_TEMPLATE(BM_potrf_static, type, size + 8); \
    BENCHMARK_TEMPLATE(BM_potrf_static, type, size + 9); \


namespace blazefeo :: benchmark
{
    template <typename Real, size_t M>
    static void BM_potrf_static(::benchmark::State& state)
    {
        StaticPanelMatrix<Real, M, M, rowMajor> A, L;
        makePositiveDefinite(A);

        for (auto _ : state)
        {
            potrf(A, L);
            DoNotOptimize(A);
            DoNotOptimize(L);
        }

        setCounters(state.counters, complexityPotrf(M, M));
        state.counters["m"] = M;
    }


    BENCHMARK_POTRF_STATIC_10(double, 1);
    BENCHMARK_POTRF_STATIC_10(double, 11);
    BENCHMARK_POTRF_STATIC_10(double, 21);
    BENCHMARK_POTRF_STATIC_10(double, 31);
    BENCHMARK_POTRF_STATIC_10(double, 41);
    BENCHMARK_POTRF_STATIC_10(double, 51);
    BENCHMARK_POTRF_STATIC_10(double, 61);
    BENCHMARK_POTRF_STATIC_10(double, 71);
    BENCHMARK_POTRF_STATIC_10(double, 81);
    BENCHMARK_POTRF_STATIC_10(double, 91);
    BENCHMARK_POTRF_STATIC_10(double, 101);
    BENCHMARK_POTRF_STATIC_10(double, 111);
    BENCHMARK_POTRF_STATIC_10(double, 121);
    BENCHMARK_POTRF_STATIC_10(double, 131);
    BENCHMARK_POTRF_STATIC_10(double, 141);
    BENCHMARK_POTRF_STATIC_10(double, 151);
    BENCHMARK_POTRF_STATIC_10(double, 161);
    BENCHMARK_POTRF_STATIC_10(double, 171);
    BENCHMARK_POTRF_STATIC_10(double, 181);
    BENCHMARK_POTRF_STATIC_10(double, 191);
    BENCHMARK_POTRF_STATIC_10(double, 201);
    BENCHMARK_POTRF_STATIC_10(double, 211);
    BENCHMARK_POTRF_STATIC_10(double, 221);
    BENCHMARK_POTRF_STATIC_10(double, 231);
    BENCHMARK_POTRF_STATIC_10(double, 241);
    BENCHMARK_POTRF_STATIC_10(double, 251);
    BENCHMARK_POTRF_STATIC_10(double, 261);
    BENCHMARK_POTRF_STATIC_10(double, 271);
    BENCHMARK_POTRF_STATIC_10(double, 281);
    BENCHMARK_POTRF_STATIC_10(double, 291);
}
