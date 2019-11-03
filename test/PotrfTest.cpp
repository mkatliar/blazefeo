#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/panel/Potrf.hpp>
#include <blazefeo/math/panel/Gemm.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

namespace blazefeo :: testing
{
    TEST(PotrfTest, test4)
    {
        size_t const M = 4;

        // Init matrices
        //
        StaticPanelMatrix<double, M, M> A, L, A1;
        makePositiveDefinite(A);
        
        // Do potrf
        potrf(A, L);

        // Check result
        A1 = 0.;
        gemm_nt(L, L, A1, A1);

        BLAZEFEO_EXPECT_APPROX_EQ(A1, A, 1e-14, 1e-14);
    }


    TEST(PotrfTest, test8)
    {
        size_t const M = 8;

        // Init matrices
        //
        StaticMatrix<double, M, M, columnMajor> blaze_A, blaze_L;
        makePositiveDefinite(blaze_A);
        llh(blaze_A, blaze_L);

        StaticPanelMatrix<double, M, M> A, L, A1;
        A.pack(data(blaze_A), spacing(blaze_A));
        
        // Do potrf
        potrf(A, L);

        // Check result
        A1 = 0.;
        gemm_nt(L, L, A1, A1);

        // std::cout << "L=\n" << L << std::endl;
        // std::cout << "blaze_L=\n" << blaze_L << std::endl;

        BLAZEFEO_EXPECT_APPROX_EQ(A1, A, 1e-14, 1e-14);
    }
}