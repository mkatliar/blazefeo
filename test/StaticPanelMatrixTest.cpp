#include <smoke/StaticPanelMatrix.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


namespace smoke :: testing
{
    TEST(StaticPanelMatrixTest, testPanelRows)
    {
        EXPECT_EQ((StaticPanelMatrix<double, 5, 7, 4>().panelRows()), 2);
        EXPECT_EQ((StaticPanelMatrix<double, 8, 7, 4>().panelRows()), 2);
    }


    TEST(StaticPanelMatrixTest, testPanelColumns)
    {
        EXPECT_EQ((StaticPanelMatrix<double, 5, 2, 4>().panelColumns()), 1);
        EXPECT_EQ((StaticPanelMatrix<double, 5, 7, 4>().panelColumns()), 2);
        EXPECT_EQ((StaticPanelMatrix<double, 5, 8, 4>().panelRows()), 2);
    }


    TEST(StaticPanelMatrixTest, testElementAccess)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;
        size_t constexpr P = 4;

        blaze::StaticMatrix<double, M, N> A_ref;
        randomize(A_ref);

        StaticPanelMatrix<double, M, N, P> A;
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                A(i, j) = A_ref(i, j);

        auto const& A_cref = A;

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
            {
                EXPECT_EQ(A(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
                EXPECT_EQ(A_cref(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
            }
    }


    TEST(StaticPanelMatrixTest, testPack)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;
        size_t constexpr P = 4;

        blaze::StaticMatrix<double, M, N, blaze::columnMajor> A_ref;
        randomize(A_ref);

        StaticPanelMatrix<double, M, N, P> A;
        A.pack(data(A_ref), spacing(A_ref));

        auto const& A_cref = A;

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                EXPECT_EQ(A(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TEST(StaticPanelMatrixTest, testUnpack)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;
        size_t constexpr P = 4;        

        StaticPanelMatrix<double, M, N, P> A;
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                blaze::randomize(A(i, j));

        blaze::StaticMatrix<double, M, N, blaze::columnMajor> A1;
        A.unpack(data(A1), spacing(A1));

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                EXPECT_EQ(A(i, j), A1(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TEST(StaticPanelMatrixTest, DISABLED_testPMatPMatMulAssign)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;
        size_t constexpr K = 10;
        size_t constexpr P = 4;        

        StaticPanelMatrix<double, M, K, P> A;
        StaticPanelMatrix<double, K, N, P> B;
        StaticPanelMatrix<double, M, N, P> D;

        randomize(A);
        randomize(B);

        D = A * B;

        // for (size_t i = 0; i < M; ++i)
        //     for (size_t j = 0; j < N; ++j)
        //         EXPECT_EQ(A(i, j), A1(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TEST(StaticPanelMatrixTest, testPMatTPMatMulAssign)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;
        size_t constexpr K = 10;
        size_t constexpr P = 4;        

        StaticPanelMatrix<double, M, K, P> A;
        StaticPanelMatrix<double, N, K, P> B;
        StaticPanelMatrix<double, M, N, P> D;

        randomize(A);
        randomize(B);

        // D = A * trans(B);

        // using E = decltype(A * trans(B));

        // E e(0);

        // using MT2 = decltype(A);
        // using MT3 = decltype(trans(B));
        // static_assert(IsPanelMatrix_v<MT2>);
        // static_assert(IsRowMajorMatrix_v<MT2>);
        // static_assert(IsPanelMatrix_v<MT3>);
        // static_assert(IsRowMajorMatrix_v<MT3>);

        // for (size_t i = 0; i < M; ++i)
        //     for (size_t j = 0; j < N; ++j)
        //         EXPECT_EQ(A(i, j), A1(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }
}