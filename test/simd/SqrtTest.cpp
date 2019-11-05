#include <test/Testing.hpp>
#include <test/Randomize.hpp>

#include <cmath>
#include <iostream>

using namespace blaze;


namespace blazefeo :: testing
{
    TEST(SqrtTest, testSqrtDiv)
    {
        std::array<double, 4> a {1., 2., 3., 4.};
        double b = 42.;
        blaze::randomize(b);
        b = -std::abs(b);
        
        __m256d x = _mm256_set_pd(a[0], a[1], a[2], a[3]);
        x *= 1. / std::sqrt(b);
        // x = _mm256_div_pd(x, _mm256_broadcastsd_pd(_mm_sqrt_sd(_mm_set_pd1(b), _mm_set_pd1(b))));

        std::cout << x[0] << ", " << x[1] << ", " << x[2] << ", " << x[3] << std::endl;
    }
}