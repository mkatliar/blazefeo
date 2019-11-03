#pragma once

#include <blaze/util/Types.h>
#include <blaze/system/Inline.h>


namespace blazefeo
{
    using namespace blaze;

    
    template <typename T, size_t M, size_t N, size_t BS>
    class GemmKernel;


    template <typename Ker>
    struct GemmKernelTraits;


    template <typename T, size_t M, size_t N, size_t BS>
    struct GemmKernelTraits<GemmKernel<T, M, N, BS>>
    {
        static size_t constexpr alignment = GemmKernel<T, M, N, BS>::alignment;
        static size_t constexpr blockSize = BS;
        static size_t constexpr blockRows = M;
        static size_t constexpr blockColumns = N;
        static size_t constexpr rows = M * BS;
        static size_t constexpr columns = N * BS;
        static size_t constexpr elementCount = rows * columns;
        static size_t constexpr blockElementCount = BS * BS;
        
        using ElementType = T;
    };


    template <bool LeftSide, bool Upper, bool TransA, typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void trsm(GemmKernel<T, M, N, BS>& ker, T const * a, T * x)
    {
        ker.template trsm<LeftSide, Upper, TransA>(a, x);
    }


    /// @brief Rank-1 update
    template <bool TA, bool TB, typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void ger(GemmKernel<T, M, N, BS>& ker, T alpha, T const * a, size_t sa, T const * b, size_t sb)
    {
        ker.template ger<TA, TB>(alpha, a, sa, b, sb);
    }


    /// @brief Rank-1 update of specified size
    template <bool TA, bool TB, typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void ger(GemmKernel<T, M, N, BS>& ker, T alpha, T const * a, size_t sa, T const * b, size_t sb, size_t m, size_t n)
    {
        ker.template ger<TA, TB>(alpha, a, sa, b, sb, m, n);
    }


    template <typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void load(GemmKernel<T, M, N, BS>& ker, T const * a, size_t sa)
    {
        ker.load(1.0, a, sa);
    }


    template <typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void load(GemmKernel<T, M, N, BS>& ker, T const * a, size_t sa, size_t m, size_t n)
    {
        ker.load(1.0, a, sa, m, n);
    }


    template <typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void load(GemmKernel<T, M, N, BS>& ker, T beta, T const * a, size_t sa)
    {
        ker.load(beta, a, sa);
    }


    template <typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void load(GemmKernel<T, M, N, BS>& ker, T beta, T const * a, size_t sa, size_t m, size_t n)
    {
        ker.load(beta, a, sa, m, n);
    }


    template <typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void store(GemmKernel<T, M, N, BS> const& ker, T * a, size_t sa)
    {
        ker.store(a, sa);
    }


    template <typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void store(GemmKernel<T, M, N, BS> const& ker, T * a, size_t sa, size_t m, size_t n)
    {
        ker.store(a, sa, m, n);
    }
}