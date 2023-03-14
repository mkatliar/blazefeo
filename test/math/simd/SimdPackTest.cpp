// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/simd/Avx256.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

#include <array>
#include <algorithm>


using namespace blaze;


namespace blazefeo :: testing
{
    template <typename T>
    class SimdPackTest
    :   public Test
    {
    };


    using MyTypes = Types<double, float>;
    TYPED_TEST_SUITE(SimdPackTest, MyTypes);


    TYPED_TEST(SimdPackTest, testHmax)
    {
        using Scalar = TypeParam;
        size_t constexpr SS = SimdPack<Scalar>::size();

        std::array<Scalar, SS> a;
        for (int i = 0; i < 1000; ++i)
        {
            randomize(a);
            SimdPack<Scalar> v {data(a), false};
            ASSERT_EQ(max(v), *std::max_element(a.begin(), a.end()));
        }
    }


    TYPED_TEST(SimdPackTest, testAbs)
    {
        using Scalar = TypeParam;
        size_t constexpr SS = SimdPack<Scalar>::size();

        blaze::StaticVector<Scalar, SS> a;
        randomize(a);
        a -= 0.5;

        SimdPack<Scalar> const v {a.data(), false};
        SimdPack<Scalar> abs_v = abs(v);

        for (size_t i = 0; i < SS; ++i)
            ASSERT_EQ(abs_v[i], std::abs(a[i]));
    }
}