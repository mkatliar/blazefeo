#include <blazefeo/math/panel/register_matrix/double_1_1_4.hpp>
#include <blazefeo/math/panel/register_matrix/double_2_1_4.hpp>
#include <blazefeo/math/panel/register_matrix/double_3_1_4.hpp>
#include <blazefeo/math/StaticPanelMatrix.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


namespace blazefeo :: testing
{
    template <typename Ker>
    class RegisterMatrixTest
    :   public Test
    {
    };


    TYPED_TEST_SUITE_P(RegisterMatrixTest);


    TYPED_TEST_P(RegisterMatrixTest, testLoadStore)
    {
        using Traits = RegisterMatrixTraits<TypeParam>;

        blaze::StaticMatrix<double, Traits::rows, Traits::columns, blaze::columnMajor> A_ref;
        randomize(A_ref);

        StaticPanelMatrix<double, Traits::rows, Traits::columns, rowMajor> A, B;
        A.pack(data(A_ref), spacing(A_ref));

        TypeParam ker;
        load(ker, A.tile(0, 0), A.spacing());
        store(ker, B.tile(0, 0), B.spacing());

        for (size_t i = 0; i < Traits::rows; ++i)
            for (size_t j = 0; j < Traits::columns; ++j)
                EXPECT_EQ(B(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TYPED_TEST_P(RegisterMatrixTest, testStore)
    {
        using Traits = RegisterMatrixTraits<TypeParam>;

        blaze::StaticMatrix<double, Traits::rows, Traits::columns, blaze::columnMajor> A_ref;
        randomize(A_ref);

        StaticPanelMatrix<double, Traits::rows, Traits::columns, rowMajor> A, B;
        A.pack(data(A_ref), spacing(A_ref));

        TypeParam ker;
        load(ker, A.tile(0, 0), A.spacing());

        for (size_t m = 0; m <= Traits::rows; ++m)
            for (size_t n = 0; n <= Traits::columns; ++n)
            {
                B = 0.;
                store(ker, B.tile(0, 0), B.spacing(), m, n);

                for (size_t i = 0; i < Traits::rows; ++i)
                    for (size_t j = 0; j < Traits::columns; ++j)
                        ASSERT_EQ(B(i, j), i < m && j < n ? A_ref(i, j) : 0.) << "element mismatch at (" << i << ", " << j << "), " 
                            << "store size = " << m << "x" << n;
            }
    }


    TYPED_TEST_P(RegisterMatrixTest, testGerNT)
    {
        using Traits = RegisterMatrixTraits<TypeParam>;

        blaze::DynamicMatrix<double, blaze::columnMajor> ma(Traits::rows, 1);
        blaze::DynamicMatrix<double, blaze::columnMajor> mb(Traits::columns, 1);
        blaze::StaticMatrix<double, Traits::rows, Traits::columns, blaze::columnMajor> mc, md;

        randomize(ma);
        randomize(mb);
        randomize(mc);

        StaticPanelMatrix<double, Traits::rows, 1, rowMajor> a;
        StaticPanelMatrix<double, Traits::columns, 1, rowMajor> b;
        StaticPanelMatrix<double, Traits::rows, Traits::columns, rowMajor> c, d;

        a.pack(data(ma), spacing(ma));
        b.pack(data(mb), spacing(mb));
        c.pack(data(mc), spacing(mc));

        TypeParam ker;
        load(ker, c.tile(0, 0), c.spacing());
        ger<false, true>(ker, 1.0, a.tile(0, 0), a.spacing(), b.tile(0, 0), b.spacing());
        store(ker, d.tile(0, 0), d.spacing());
        
        d.unpack(data(md), spacing(md));

        BLAZEFEO_EXPECT_EQ(md, evaluate(mc + ma * trans(mb)));
    }


    TYPED_TEST_P(RegisterMatrixTest, testPotrf)
    {
        using Traits = RegisterMatrixTraits<TypeParam>;
        TypeParam ker;

        if constexpr (Traits::rows == Traits::columns)
        {
            StaticPanelMatrix<typename Traits::ElementType, Traits::rows, Traits::columns, rowMajor> A, B, A1;
            makePositiveDefinite(A);

            load(ker, A.tile(0, 0), A.spacing());
            ker.potrf();
            store(ker, B.tile(0, 0), B.spacing());
            
            A1 = 0.;
            gemm_nt(B, B, A1, A1);

            BLAZEFEO_ASSERT_APPROX_EQ(A1, A, 1e-15, 1e-15);
        }
        else
        {
            std::clog << "RegisterMatrixTest.testPotrf not implemented for non-square kernels!" << std::endl;
        }        
    }


    TYPED_TEST_P(RegisterMatrixTest, testTrsmRLT)
    {
        using Traits = RegisterMatrixTraits<TypeParam>;
        TypeParam ker;

        if constexpr (Traits::rows == Traits::columns)
        {
            using blaze::randomize;
            StaticPanelMatrix<typename Traits::ElementType, Traits::rows, Traits::columns, rowMajor> L, A, X, A1;            
            
            for (size_t i = 0; i < Traits::rows; ++i)
                for (size_t j = 0; j < Traits::columns; ++j)
                    if (j <= i)
                        randomize(L(i, j));
                    else
                        reset(L(i, j));
            
            randomize(A);

            load(ker, L.tile(0, 0), L.spacing());
            trsm<false, false, true>(ker, A.tile(0, 0), X.tile(0, 0));

            A1 = 0.;
            gemm_nt(X, L, A1, A1);

            std::cout << A << std::endl;
            std::cout << A1 << std::endl;

            BLAZEFEO_ASSERT_APPROX_EQ(A1, A, 1e-14, 1e-14);
        }
        else
        {
            std::clog << "RegisterMatrixTest.testTrsmRLT not implemented for non-square kernels!" << std::endl;
        }        
    }


    REGISTER_TYPED_TEST_SUITE_P(RegisterMatrixTest,
        testLoadStore,
        testStore,
        testGerNT,
        testPotrf,
        testTrsmRLT
    );


    using RegisterMatrix_double_1_1_4 = RegisterMatrix<double, 1, 1, 4>;
    using RegisterMatrix_double_2_1_4 = RegisterMatrix<double, 2, 1, 4>;
    using RegisterMatrix_double_3_1_4 = RegisterMatrix<double, 3, 1, 4>;

    INSTANTIATE_TYPED_TEST_SUITE_P(RegisterMatrix_double_1_1_4, RegisterMatrixTest, RegisterMatrix_double_1_1_4);
    INSTANTIATE_TYPED_TEST_SUITE_P(RegisterMatrix_double_2_1_4, RegisterMatrixTest, RegisterMatrix_double_2_1_4);
    INSTANTIATE_TYPED_TEST_SUITE_P(RegisterMatrix_double_3_1_4, RegisterMatrixTest, RegisterMatrix_double_3_1_4);


    TEST(RegisterMatrix_double_1_1_4_Test, testPotrf)
    {
        RegisterMatrix<double, 1, 1, 4> ker;

        StaticPanelMatrix<double, 4, 4, rowMajor> A {
            {4,     6,    10,    16},
            {6,    25,    39,    60},
            {10,    39,   110,   164},
            {16,    60,   164,   366},
        };

        std::cout << A << std::endl;

        load(ker, A.tile(0, 0), A.spacing());
        ker.potrf();
        store(ker, A.tile(0, 0), A.spacing());

        std::cout << A << std::endl;
    }


    TEST(RegisterMatrix_double_1_1_4_Test, testTrsmRLT)
    {
        RegisterMatrix<double, 1, 1, 4> ker;

        StaticPanelMatrix<double, 4, 4, rowMajor> L {
            {2,            0,            0,            0},
            {3,            4,            0,            0},
            {5,            6,            7,            0},
            {8,            9,           10,           11},
        };

        StaticPanelMatrix<double, 4, 4, rowMajor> A {
            {1,   0,   0,   0},
            {0,   1,   0,   0},
            {0,   0,   1,   0},
            {0,   0,   0,   1},
        };

        load(ker, L.tile(0, 0), L.spacing());
        trsm<false, false, true>(ker, tile(A, 0, 0), tile(A, 0, 0));

        std::cout << A << std::endl;
    }
}