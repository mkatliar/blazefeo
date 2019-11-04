#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/views/submatrix/Panel.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


namespace blazefeo :: testing
{
    TEST(SubmatrixTest, testSubmatrixOfStaticPanelMatrix)
    {
        StaticPanelMatrix<double, 12, 12, rowMajor> A;
        auto B = submatrix(A, 4, 0, 8, 8);
        
        // PanelSubmatrix<decltype(A), rowMajor> B(A, 4, 0, 8, 8);
        std::cout << B << std::endl;
    }


    TEST(SubmatrixTest, testSubmatrixOfConstStaticPanelMatrix)
    {
        StaticPanelMatrix<double, 12, 12, rowMajor> const A;
        auto B = submatrix(A, 4, 0, 8, 8);
        
        // PanelSubmatrix<decltype(A), rowMajor> B(A, 4, 0, 8, 8);
        std::cout << B << std::endl;
    }


    TEST(SubmatrixTest, testSubmatrixOfDynamicPanelMatrix)
    {
        DynamicPanelMatrix<double, rowMajor> A(12, 12);
        auto B = submatrix(A, 4, 0, 8, 8);

        static_assert(std::is_same_v<decltype(tile(B, 0, 0)), double *>);
        tile(B, 0, 0);
        
        // PanelSubmatrix<decltype(A), rowMajor> B(A, 4, 0, 8, 8);
        std::cout << B << std::endl;
    }


    TEST(SubmatrixTest, testSubmatrixOfConstDynamicPanelMatrix)
    {
        DynamicPanelMatrix<double, rowMajor> const A(12, 12);
        auto B = submatrix(A, 4, 0, 8, 8);

        static_assert(std::is_same_v<decltype(tile(B, 0, 0)), double const *>);
        tile(B, 0, 0);
        
        // PanelSubmatrix<decltype(A), rowMajor> B(A, 4, 0, 8, 8);
        std::cout << B << std::endl;
    }


    TEST(SubmatrixTest, testTile)
    {
        DynamicPanelMatrix<double, rowMajor> A(12, 12);
        A = 0.;
        auto B = submatrix(A, 4, 0, 8, 8);

        *tile(B, 0, 0) = 1.;
        *tile(B, 1, 0) = 2.;
        *tile(B, 0, 1) = 3.;
        *tile(B, 1, 1) = 4.;
        
        EXPECT_EQ(A(4, 0), 1.);
        EXPECT_EQ(A(8, 0), 2.);
        EXPECT_EQ(A(4, 4), 3.);
        EXPECT_EQ(A(8, 4), 4.);
    }
}