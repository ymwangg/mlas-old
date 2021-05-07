#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "mlas.h"

inline void
CPUCacheFlushImpl(const char* addr, unsigned int len)
{
// TODO(FrozenGene): Support ARM.
#if (defined(_M_X64) || defined(__x86_64__))
    const size_t cache_line = 64;
    if (addr == nullptr || len <= 0) {
        return;
    }

    for (uintptr_t uptr = (uintptr_t)addr & ~(cache_line - 1); uptr < (uintptr_t)addr + len;
         uptr += cache_line) {
        _mm_clflush(reinterpret_cast<const void*>(uptr));
    }

#endif
}

void
flush(float* a, float* b, int m, int k, int n)
{
    for (int i = 0; i < m * k; i++) {
        a[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    for (int i = 0; i < k * n; i++) {
        b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

void
SGEMM(bool pack_b,
      bool trans_a,
      bool trans_b,
      size_t M,
      size_t N,
      size_t K,
      float alpha = 1.0f,
      float beta = 0.0f)
{
    std::vector<float> A(static_cast<size_t>(M * K));
    std::vector<float> B(static_cast<size_t>(K * N));
    std::vector<float> C(static_cast<size_t>(M * N));
    printf("m=%ld n=%ld k=%ld\n", M, N, K);
    double tsum = 0;
    const int iter = 1000;
    if (pack_b) {
        size_t pack_b_size = MlasGemmPackBSize(N, K);
        printf("pack_b_size=%ld\n", pack_b_size);
        std::vector<float> B_packed(pack_b_size);
        MlasGemmPackB(CblasNoTrans, N, K, B.data(), N, B_packed.data());

        MlasGemm(trans_a ? CblasTrans : CblasNoTrans, static_cast<size_t>(M),
                 static_cast<size_t>(N), static_cast<size_t>(K), alpha, A.data(), trans_a ? M : K,
                 B_packed.data(), beta, C.data(), N);

        for (int i = 0; i < iter; i++) {
            CPUCacheFlushImpl((char*)A.data(), M * K * sizeof(float));
            CPUCacheFlushImpl((char*)B.data(), K * N * sizeof(float));
            auto t1 = std::chrono::high_resolution_clock::now();
            MlasGemm(trans_a ? CblasTrans : CblasNoTrans, static_cast<size_t>(M),
                     static_cast<size_t>(N), static_cast<size_t>(K), alpha, A.data(),
                     trans_a ? M : K, B_packed.data(), beta, C.data(), N);
            auto t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> dt = t2 - t1;
            tsum += dt.count();
        }

    } else {
        MlasGemm(trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
                 static_cast<size_t>(M), static_cast<size_t>(N), static_cast<size_t>(K), alpha,
                 A.data(), trans_a ? M : K, B.data(), trans_b ? K : N, beta, C.data(), N);

        for (int i = 0; i < iter; i++) {
            CPUCacheFlushImpl((char*)A.data(), M * K * sizeof(float));
            CPUCacheFlushImpl((char*)B.data(), K * N * sizeof(float));
            auto t1 = std::chrono::high_resolution_clock::now();
            MlasGemm(trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
                     static_cast<size_t>(M), static_cast<size_t>(N), static_cast<size_t>(K), alpha,
                     A.data(), trans_a ? M : K, B.data(), trans_b ? K : N, beta, C.data(), N);
            auto t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> dt = t2 - t1;
            tsum += dt.count();
        }
    }
    printf("tsum = %f ms\n", tsum / iter * 1000);
}

int
main(int argc, char** argv)
{
    // const int iter = 1000;
    // int m = atoi(argv[1]);
    // int k = atoi(argv[2]);
    // int n = atoi(argv[3]);
    // std::string do_flush;
    // if (argc == 5) {
    //     do_flush = argv[4];
    // }

    SGEMM(true, false, false, 64, 3072, 768);
    return 0;
}
