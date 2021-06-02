#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "mlas.h"

#include <stdio.h>
#include <memory.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <malloc.h>
template <typename T>
class MatrixGuardBuffer {
 public:
  MatrixGuardBuffer() {
    _BaseBuffer = nullptr;
    _BaseBufferSize = 0;
    _ElementsAllocated = 0;
  }

  ~MatrixGuardBuffer(void) {
    ReleaseBuffer();
  }

  T* GetBuffer(size_t Elements, bool ZeroFill = false) {
    //
    // Check if the internal buffer needs to be reallocated.
    //

    if (Elements > _ElementsAllocated) {
      ReleaseBuffer();

      //
      // Reserve a virtual address range for the allocation plus an unmapped
      // guard region.
      //

      constexpr size_t BufferAlignment = 64 * 1024;
      constexpr size_t GuardPadding = 256 * 1024;

      size_t BytesToAllocate = ((Elements * sizeof(T)) + BufferAlignment - 1) & ~(BufferAlignment - 1);

      _BaseBufferSize = BytesToAllocate + GuardPadding;

#if defined(_WIN32)
      _BaseBuffer = VirtualAlloc(NULL, _BaseBufferSize, MEM_RESERVE, PAGE_NOACCESS);
#else
      _BaseBuffer = mmap(0, _BaseBufferSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

      if (_BaseBuffer == nullptr) {
        abort();
      }

      //
      // Commit the number of bytes for the allocation leaving the upper
      // guard region as unmapped.
      //

#if defined(_WIN32)
      if (VirtualAlloc(_BaseBuffer, BytesToAllocate, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
        ORT_THROW_EX(std::bad_alloc);
      }
#else
      if (mprotect(_BaseBuffer, BytesToAllocate, PROT_READ | PROT_WRITE) != 0) {
        abort();
      }
#endif

      _ElementsAllocated = BytesToAllocate / sizeof(T);
      _GuardAddress = (T*)((unsigned char*)_BaseBuffer + BytesToAllocate);
    }

    //
    //
    //

    T* GuardAddress = _GuardAddress;
    T* buffer = GuardAddress - Elements;

    if (ZeroFill) {
      std::fill_n(buffer, Elements, T(0));

    } else {
      const int MinimumFillValue = -23;
      const int MaximumFillValue = 23;

      int FillValue = MinimumFillValue;
      T* FillAddress = buffer;

      while (FillAddress < GuardAddress) {
        *FillAddress++ = (T)FillValue;

        FillValue++;

        if (FillValue > MaximumFillValue) {
          FillValue = MinimumFillValue;
        }
      }
    }

    return buffer;
  }

  void ReleaseBuffer(void) {
    if (_BaseBuffer != nullptr) {
#if defined(_WIN32)
      VirtualFree(_BaseBuffer, 0, MEM_RELEASE);
#else
      munmap(_BaseBuffer, _BaseBufferSize);
#endif

      _BaseBuffer = nullptr;
      _BaseBufferSize = 0;
    }

    _ElementsAllocated = 0;
  }

 private:
  size_t _ElementsAllocated;
  void* _BaseBuffer;
  size_t _BaseBufferSize;
  T* _GuardAddress;
};


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
        MatrixGuardBuffer<uint8_t> BufferBPacked;
        void* PackedB = BufferBPacked.GetBuffer(pack_b_size, true);
        // void* PackedB2 = memalign(64, pack_b_size);
        void* PackedB2 = memalign(32, pack_b_size);
        std::cout << A.data() << std::endl;
        std::cout << PackedB2 << std::endl;
        // void* PackedB2 = malloc(pack_b_size);

        for (int i = 0; i < A.size(); i++) {
            A[i] = i;
        }
        for (int i = 0; i < B.size(); i++) {
            B[i] = i;
        }
        MlasGemmPackB(CblasNoTrans, N, K, B.data(), N, PackedB);
        MlasGemmPackB(CblasNoTrans, N, K, B.data(), N, PackedB2);
        float* new_b = (float*)PackedB2;
        for (int i = 0; i < pack_b_size/4; i++) {
            std::cout << new_b[i] << " ";
        }
        std::cout << std::endl;

        MlasGemm(trans_a ? CblasTrans : CblasNoTrans, static_cast<size_t>(M),
                 static_cast<size_t>(N), static_cast<size_t>(K), alpha, A.data(), trans_a ? M : K,
                 PackedB2, beta, C.data(), N);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << C[i*N + j] << " ";
            }
            std::cout << std::endl;
        }

        for (int i = 0; i < iter; i++) {
            CPUCacheFlushImpl((char*)A.data(), M * K * sizeof(float));
            CPUCacheFlushImpl((char*)PackedB, pack_b_size);
            auto t1 = std::chrono::high_resolution_clock::now();
            MlasGemm(trans_a ? CblasTrans : CblasNoTrans, static_cast<size_t>(M),
                     static_cast<size_t>(N), static_cast<size_t>(K), alpha, A.data(),
                     trans_a ? M : K, PackedB, beta, C.data(), N);
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
    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    // std::string do_flush;
    // if (argc == 5) {
    //     do_flush = argv[4];
    // }

    SGEMM(true, false, false, m, n, k);
    return 0;
}
