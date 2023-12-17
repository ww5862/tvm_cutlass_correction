#include<iostream>
#include<cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include <cutlass/util/host_tensor.h>

#include<vector>

void cutlass_strided_bathed_sgemm(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm50,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
                                                    cutlass::gemm::GemmShape<8, 16, 8>,
                                                    cutlass::gemm::GemmShape<1, 1, 1>,
                                                    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
                                                    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
                                                    2,
                                                    1,
                                                    1,
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
     9};
    
    size_t workspace_size = gemm_op.get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());

    if(status == cutlass::Status::kSuccess){
        std::cout << "Passed" << std::endl;
    }
    else{
        std::cout << "Failed" << std::endl;
    }

    status = gemm_op();

    cudaDeviceSynchronize();

    std::cout << "done" << std::endl;

    if(status == cutlass::Status::kSuccess){
        std::cout << "Passed" << std::endl;
    }
    else{
        std::cout << "Failed" << std::endl;
    }

    int const count_C = batch_count * ldc * m;
    std::vector<float> host_C(count_C);

    cudaMemcpy(host_C.data(), C, count_C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemset(C, 0, count_C * sizeof(float));

    using Gemm2 = cutlass::gemm::device::GemmBatched<float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm50,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
                                                    cutlass::gemm::GemmShape<8, 16, 8>,
                                                    cutlass::gemm::GemmShape<1, 1, 1>,
                                                    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
                                                    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
                                                    2,
                                                    1,
                                                    1,
                                                    false,
                                                    cutlass::arch::OpMultiplyAdd
                                                    >;
    Gemm2 gemm2_op;

    gemm2_op({
     {m, n, k},
     {A, lda}, batch_stride_A,
     {B, ldb}, batch_stride_B,
     {C, ldc}, batch_stride_C,
     {C, ldc}, batch_stride_C,
     {alpha, beta},
     batch_count});

    std::vector<float> host_C2(count_C);
    cudaMemcpy(host_C2.data(), C, count_C * sizeof(float), cudaMemcpyDeviceToHost);
    

    for(int i = 0; i < host_C.size(); i++){
        if(host_C[i] - host_C2[i] > 10e-9){
            std::cout << "false" << std::endl;
            std::cout << host_C[i] << " " << host_C2[i] << std::endl;
            break;
        }
        // std::cout << host_C[i] << " " << host_C2[i] << std::endl;

    }


}

int main(int argc, char *argv[]){
    int tmp_m = 64;
    int tmp_n = 128;
    int tmp_k = 64;
    int tmp_b = 10;

    //define dimension
    int const m = tmp_m;
    int const n = tmp_n;
    int const k = tmp_k;
    int const batch_count = tmp_b;

    //NT -> T
    int const lda = m;
    int const ldb = n;
    int const ldc = n;

    int const count_A = batch_count * lda * k;
    int const count_B = batch_count * ldb * k;
    int const count_C = batch_count * ldc * m;

    long long int batch_stride_A = static_cast<long long int>(lda) * static_cast<long long int>(k);
    long long int batch_stride_B = static_cast<long long int>(ldb) * static_cast<long long int>(k);
    long long int batch_stride_C = static_cast<long long int>(ldc) * static_cast<long long int>(m);

    float alpha = 1.0f;
    float beta = 0.0f;

    cudaError_t result = cudaSuccess;

    std::vector<float> host_A(count_A);
    std::vector<float> host_B(count_B);
    std::vector<float> host_C(count_C);

    srand(time(NULL));

    for(int i = 0; i < host_A.size(); i++) host_A[i] = rand() % 100;
    for(int i = 0; i < host_B.size(); i++) host_B[i] = rand() % 100;

    float *A;
    float *B;
    float *C;

    result = cudaMalloc(&A, count_A * sizeof(float));
    result = cudaMalloc(&B, count_B * sizeof(float));
    result = cudaMalloc(&C, count_C * sizeof(float));

    result = cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice);
    result = cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice);
    result = cudaMemcpy(C, host_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice);

    if(result != cudaSuccess) std::cout << "not success" << std::endl;

    cutlass_strided_bathed_sgemm(m, n, k, alpha, A, lda, batch_stride_A, B,
                                           ldb, batch_stride_B, C, ldc, batch_stride_C,
                                           beta, batch_count);

    // result = cudaMemcpy(host_C.data(), C, count_C * sizeof(float), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < host_C.size(); i++) std::cout << host_C[i] << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
