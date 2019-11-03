#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/views/submatrix/Panel.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


namespace blazefeo :: testing
{
    TEST(SubmatrixTest, testSubmatrix)
    {
        StaticPanelMatrix<double, 12, 12> A;
        auto B = submatrix(A, 4, 0, 8, 8);
        
        // PanelSubmatrix<decltype(A), rowMajor> B(A, 4, 0, 8, 8);
        std::cout << B << std::endl;
    }
}