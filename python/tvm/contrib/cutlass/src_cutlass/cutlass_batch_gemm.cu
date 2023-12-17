#include<iostream>
#include<cuda_runtime.h>
#include "utils.h"
#include "../Malloc/data.h"

#define EXPORT

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include <cutlass/util/host_tensor.h>

#pragma warning( disable : 4503)

float cutlass_strided_bathed_sgemm(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup, int repeat
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<MM, MN, MK>,
                                                    cutlass::gemm::GemmShape<WM, WN, WK>,
                                                    cutlass::gemm::GemmShape<1, 1, 1>,
                                                    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
                                                    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
                                                    KStage,
                                                    Al,
                                                    Al,
                                                    true,
                                                    cutlass::arch::OpMultiplyAdd
                                                    >;
    Gemm gemm_op;

    Gemm::Arguments arguments{
        {m, n, k},
        {A, lda}, batch_stride_A,
        {B, ldb}, batch_stride_B,
        {C, ldc}, batch_stride_C,
        {C, ldc}, batch_stride_C,
        {alpha, beta},
        batch_count,
        split_k
    };

    size_t workspace_size = gemm_op.get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());

    if(status != cutlass::Status::kSuccess) return -1;
    cudaDeviceSynchronize();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    for(int i = 0; i < warmup; i++){

        status = gemm_op();

        if(status != cutlass::Status::kSuccess){
        cudaEventDestroy(start);
        cudaEventDestroy(end);

        return -1;
        }
    }

    cudaDeviceSynchronize();

    float total_time = 0.0f;
    for(int i = 0; i < repeat; i++){
        cudaEventRecord(start);

        status = gemm_op();

        cudaEventRecord(end);
        cudaEventSynchronize(end);
        
        float time;
        cudaEventElapsedTime(&time, start, end);
        total_time += time;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return total_time / repeat;

}


extern "C"{
    EXPORT float cutlass_batch(void *A, void *B, void *C, int split_k, int warmup, int repeat){

        Data *d_A = static_cast<Data *>(A);
        Data *d_B = static_cast<Data *>(B);
        Data *d_C = static_cast<Data *>(C);

        int m = d_A->row;
        int n = d_B->column;
        int k = d_A->column;
        int batch_count = d_A->batch;

        //TN -> T
        int const lda = k;
        int const ldb = k;
        int const ldc = n;

        //NT -> T
        // int const lda = m;
        // int const ldb = n;
        // int const ldc = n;

        //TN -> T
        long long int batch_stride_A = static_cast<long long int>(lda) * static_cast<long long int>(m);
        long long int batch_stride_B = static_cast<long long int>(ldb) * static_cast<long long int>(n);
        long long int batch_stride_C = static_cast<long long int>(ldc) * static_cast<long long int>(m);

        //NT -> T
        // long long int batch_stride_A = static_cast<long long int>(m) * static_cast<long long int>(k);
        // long long int batch_stride_B = static_cast<long long int>(n) * static_cast<long long int>(k);
        // long long int batch_stride_C = static_cast<long long int>(n) * static_cast<long long int>(m);

        float alpha = 1.0f;
        float beta = 0.0f;

        float *gpu_a = static_cast<float *>(d_A->p);
        float *gpu_b = static_cast<float *>(d_B->p);
        float *gpu_c = static_cast<float *>(d_C->p);

        float result = cutlass_strided_bathed_sgemm(m, n, k, alpha, gpu_a, lda, batch_stride_A, gpu_b, ldb, batch_stride_B, gpu_c, ldc, batch_stride_C, beta, batch_count, split_k, warmup, repeat);

        // cudaDeviceReset();
        return result;  
    }
}