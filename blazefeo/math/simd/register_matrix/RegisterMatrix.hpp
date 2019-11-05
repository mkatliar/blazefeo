#pragma once

#include <blazefeo/math/simd/Simd.hpp>

#include <blaze/util/Types.h>
#include <blaze/system/Inline.h>

#include <immintrin.h>


namespace blazefeo
{
    using namespace blaze;


    template <typename T, size_t M, size_t N, size_t SS>
    class RegisterMatrix
    {
    public:
        /// @brief Default ctor
        RegisterMatrix()
        {
        }


        /// @brief load from memory
        void load(T beta, T const * ptr, size_t spacing);


        /// @brief load from memory with specified size
        void load(T beta, T const * ptr, size_t spacing, size_t m, size_t n);


        /// @brief store to memory
        void store(T * ptr, size_t spacing) const;


        /// @brief store to memory with specified size
        void store(T * ptr, size_t spacing, size_t m, size_t n) const;


        /// @brief Rank-1 update
        template <bool TA, bool TB>
        void ger(T alpha, T const * a, size_t sa, T const * b, size_t sb);


        /// @brief Rank-1 update of specified size
        template <bool TA, bool TB>
        void ger(T alpha, T const * a, size_t sa, T const * b, size_t sb, size_t m, size_t n);


        /// @brief In-place Cholesky decomposition
        void potrf();


        /// @brief Triangular substitution
        template <bool LeftSide, bool Upper, bool TransA>
        void trsm(T const * a, T * x) const;


    private:
        using IntrinsicType = typename Simd<T, SS>::IntrinsicType;
        
        IntrinsicType v_[M][N];
    };


    template <typename Ker>
    struct RegisterMatrixTraits;


    template <typename T, size_t M, size_t N, size_t SS>
    struct RegisterMatrixTraits<RegisterMatrix<T, M, N, SS>>
    {
        static size_t constexpr simdSize = SS;
        static size_t constexpr rows = M * SS;
        static size_t constexpr columns = N;
        static size_t constexpr elementCount = rows * columns;
        
        using ElementType = T;
    };


    template<typename T, size_t M, size_t N, size_t SS>
    inline void RegisterMatrix<T, M, N, SS>::load(T beta, T const * ptr, size_t spacing)
    {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                v_[i][j] = blazefeo::load<SS>(ptr + spacing * i + SS * j);
    }


    template<typename T, size_t M, size_t N, size_t SS>
    inline void RegisterMatrix<T, M, N, SS>::store(T * ptr, size_t spacing) const
    {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                blazefeo::store(ptr + spacing * i + SS * j, v_[i][j]);
    }


    template <bool LeftSide, bool Upper, bool TransA, typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void trsm(RegisterMatrix<T, M, N, SS>& ker, T const * a, T * x)
    {
        ker.template trsm<LeftSide, Upper, TransA>(a, x);
    }


    /// @brief Rank-1 update
    template <bool TA, bool TB, typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void ger(RegisterMatrix<T, M, N, SS>& ker, T alpha, T const * a, size_t sa, T const * b, size_t sb)
    {
        ker.template ger<TA, TB>(alpha, a, sa, b, sb);
    }


    /// @brief Rank-1 update of specified size
    template <bool TA, bool TB, typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void ger(RegisterMatrix<T, M, N, SS>& ker, T alpha, T const * a, size_t sa, T const * b, size_t sb, size_t m, size_t n)
    {
        ker.template ger<TA, TB>(alpha, a, sa, b, sb, m, n);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void load(RegisterMatrix<T, M, N, SS>& ker, T const * a, size_t sa)
    {
        ker.load(1.0, a, sa);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void load(RegisterMatrix<T, M, N, SS>& ker, T const * a, size_t sa, size_t m, size_t n)
    {
        ker.load(1.0, a, sa, m, n);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void load(RegisterMatrix<T, M, N, SS>& ker, T beta, T const * a, size_t sa)
    {
        ker.load(beta, a, sa);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void load(RegisterMatrix<T, M, N, SS>& ker, T beta, T const * a, size_t sa, size_t m, size_t n)
    {
        ker.load(beta, a, sa, m, n);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void store(RegisterMatrix<T, M, N, SS> const& ker, T * a, size_t sa)
    {
        ker.store(a, sa);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void store(RegisterMatrix<T, M, N, SS> const& ker, T * a, size_t sa, size_t m, size_t n)
    {
        ker.store(a, sa, m, n);
    }
}