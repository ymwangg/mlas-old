/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    threading.cpp

Abstract:

    This module implements platform specific threading support.

--*/

#include "mlasi.h"

void
MlasExecuteThreaded(
    MLAS_THREADED_ROUTINE* ThreadedRoutine,
    void* Context,
    ptrdiff_t Iterations
    )
{
    //
    // Execute the routine directly if only one iteration is specified.
    //

    if (Iterations == 1) {
        ThreadedRoutine(Context, 0);
        return;
    }



    //
    // Fallback to OpenMP or a serialized implementation.
    //

    //
    // Execute the routine for the specified number of iterations.
    //

#pragma omp parallel for
    for (ptrdiff_t tid = 0; tid < Iterations; tid++) {
        ThreadedRoutine(Context, tid);
    }

}


void
MlasTrySimpleParallel(
    const std::ptrdiff_t Iterations,
    const std::function<void(std::ptrdiff_t tid)>& Work)
{
    //
    // Execute the routine directly if only one iteration is specified.
    //
    if (Iterations == 1) {
        Work(0);
        return;
    }


#pragma omp parallel for
    for (ptrdiff_t tid = 0; tid < Iterations; tid++) {
        Work(tid);
    }
}
