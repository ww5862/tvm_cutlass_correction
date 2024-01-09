
#include<iostream>
#include<cuda_runtime.h>

#include <unistd.h>
#include<string>
#include<fstream>      

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include <cutlass/util/host_tensor.h>


float cutlass_strided_bathed_sgemm_0(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
                                                    cutlass::gemm::GemmShape<4, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 32, 8], [4, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 32, 8], [4, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_1(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 32, 8], [8, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 32, 8], [8, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_2(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
                                                    cutlass::gemm::GemmShape<8, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 32, 8], [8, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 32, 8], [8, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_3(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
                                                    cutlass::gemm::GemmShape<16, 8, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 32, 8], [16, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 32, 8], [16, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_4(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
                                                    cutlass::gemm::GemmShape<16, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 32, 8], [16, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 32, 8], [16, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_5(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
                                                    cutlass::gemm::GemmShape<16, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 32, 8], [16, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 32, 8], [16, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_6(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
                                                    cutlass::gemm::GemmShape<32, 4, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 32, 8], [32, 4, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 32, 8], [32, 4, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_7(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
                                                    cutlass::gemm::GemmShape<32, 8, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 32, 8], [32, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 32, 8], [32, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_8(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
                                                    cutlass::gemm::GemmShape<32, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 32, 8], [32, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 32, 8], [32, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_9(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 32, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 32, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_10(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 64, 8>,
                                                    cutlass::gemm::GemmShape<4, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 64, 8], [4, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 64, 8], [4, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_11(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 64, 8>,
                                                    cutlass::gemm::GemmShape<8, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 64, 8], [8, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 64, 8], [8, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_12(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 64, 8>,
                                                    cutlass::gemm::GemmShape<8, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 64, 8], [8, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 64, 8], [8, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_13(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 64, 8>,
                                                    cutlass::gemm::GemmShape<16, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 64, 8], [16, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 64, 8], [16, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_14(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 64, 8>,
                                                    cutlass::gemm::GemmShape<16, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 64, 8], [16, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 64, 8], [16, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_15(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 64, 8>,
                                                    cutlass::gemm::GemmShape<16, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 64, 8], [16, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 64, 8], [16, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_16(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 64, 8>,
                                                    cutlass::gemm::GemmShape<32, 8, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 64, 8], [32, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 64, 8], [32, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_17(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 64, 8>,
                                                    cutlass::gemm::GemmShape<32, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 64, 8], [32, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 64, 8], [32, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_18(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 64, 8>,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 64, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 64, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_19(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 96, 8>,
                                                    cutlass::gemm::GemmShape<16, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 96, 8], [16, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 96, 8], [16, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_20(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 96, 8>,
                                                    cutlass::gemm::GemmShape<32, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 96, 8], [32, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 96, 8], [32, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_21(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 96, 8>,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 96, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 96, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_22(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 128, 8>,
                                                    cutlass::gemm::GemmShape<4, 128, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 128, 8], [4, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 128, 8], [4, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_23(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 128, 8>,
                                                    cutlass::gemm::GemmShape<8, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 128, 8], [8, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 128, 8], [8, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_24(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 128, 8>,
                                                    cutlass::gemm::GemmShape<8, 128, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 128, 8], [8, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 128, 8], [8, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_25(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 128, 8>,
                                                    cutlass::gemm::GemmShape<16, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 128, 8], [16, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 128, 8], [16, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_26(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 128, 8>,
                                                    cutlass::gemm::GemmShape<16, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 128, 8], [16, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 128, 8], [16, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_27(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 128, 8>,
                                                    cutlass::gemm::GemmShape<32, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 128, 8], [32, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 128, 8], [32, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_28(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 128, 8>,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 128, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 128, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_29(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 160, 8>,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 160, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 160, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_30(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 192, 8>,
                                                    cutlass::gemm::GemmShape<16, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 192, 8], [16, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 192, 8], [16, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_31(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 192, 8>,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 192, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 192, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_32(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 224, 8>,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 224, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 224, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_33(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 256, 8>,
                                                    cutlass::gemm::GemmShape<4, 256, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 256, 8], [4, 256, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 256, 8], [4, 256, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_34(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 256, 8>,
                                                    cutlass::gemm::GemmShape<8, 128, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 256, 8], [8, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 256, 8], [8, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_35(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 256, 8>,
                                                    cutlass::gemm::GemmShape<16, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 256, 8], [16, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 256, 8], [16, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_36(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<32, 256, 8>,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[32, 256, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[32, 256, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_37(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
                                                    cutlass::gemm::GemmShape<8, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 32, 8], [8, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 32, 8], [8, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_38(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
                                                    cutlass::gemm::GemmShape<16, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 32, 8], [16, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 32, 8], [16, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_39(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
                                                    cutlass::gemm::GemmShape<16, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 32, 8], [16, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 32, 8], [16, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_40(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
                                                    cutlass::gemm::GemmShape<32, 8, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 32, 8], [32, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 32, 8], [32, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_41(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
                                                    cutlass::gemm::GemmShape<32, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 32, 8], [32, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 32, 8], [32, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_42(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 32, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 32, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_43(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
                                                    cutlass::gemm::GemmShape<64, 4, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 32, 8], [64, 4, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 32, 8], [64, 4, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_44(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
                                                    cutlass::gemm::GemmShape<64, 8, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 32, 8], [64, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 32, 8], [64, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_45(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
                                                    cutlass::gemm::GemmShape<64, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 32, 8], [64, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 32, 8], [64, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_46(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 32, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 32, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_47(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 64, 8>,
                                                    cutlass::gemm::GemmShape<8, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 64, 8], [8, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 64, 8], [8, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_48(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 64, 8>,
                                                    cutlass::gemm::GemmShape<16, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 64, 8], [16, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 64, 8], [16, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_49(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 64, 8>,
                                                    cutlass::gemm::GemmShape<16, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 64, 8], [16, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 64, 8], [16, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_50(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 64, 8>,
                                                    cutlass::gemm::GemmShape<32, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 64, 8], [32, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 64, 8], [32, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_51(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 64, 8>,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 64, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 64, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_52(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 64, 8>,
                                                    cutlass::gemm::GemmShape<32, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 64, 8], [32, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 64, 8], [32, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_53(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 64, 8>,
                                                    cutlass::gemm::GemmShape<64, 8, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 64, 8], [64, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 64, 8], [64, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_54(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 64, 8>,
                                                    cutlass::gemm::GemmShape<64, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 64, 8], [64, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 64, 8], [64, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_55(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 64, 8>,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 64, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 64, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_56(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 96, 8>,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 96, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 96, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_57(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 96, 8>,
                                                    cutlass::gemm::GemmShape<64, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 96, 8], [64, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 96, 8], [64, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_58(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 96, 8>,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 96, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 96, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_59(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 128, 8>,
                                                    cutlass::gemm::GemmShape<8, 128, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 128, 8], [8, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 128, 8], [8, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_60(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 128, 8>,
                                                    cutlass::gemm::GemmShape<16, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 128, 8], [16, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 128, 8], [16, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_61(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 128, 8>,
                                                    cutlass::gemm::GemmShape<16, 128, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 128, 8], [16, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 128, 8], [16, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_62(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 128, 8>,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 128, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 128, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_63(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 128, 8>,
                                                    cutlass::gemm::GemmShape<32, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 128, 8], [32, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 128, 8], [32, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_64(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 128, 8>,
                                                    cutlass::gemm::GemmShape<64, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 128, 8], [64, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 128, 8], [64, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_65(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 128, 8>,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 128, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 128, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_66(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 160, 8>,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 160, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 160, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_67(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 192, 8>,
                                                    cutlass::gemm::GemmShape<32, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 192, 8], [32, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 192, 8], [32, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_68(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 192, 8>,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 192, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 192, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_69(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 224, 8>,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 224, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 224, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_70(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 256, 8>,
                                                    cutlass::gemm::GemmShape<8, 256, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 256, 8], [8, 256, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 256, 8], [8, 256, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_71(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 256, 8>,
                                                    cutlass::gemm::GemmShape<16, 128, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 256, 8], [16, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 256, 8], [16, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_72(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 256, 8>,
                                                    cutlass::gemm::GemmShape<32, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 256, 8], [32, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 256, 8], [32, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_73(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<64, 256, 8>,
                                                    cutlass::gemm::GemmShape<64, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[64, 256, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[64, 256, 8], [64, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_74(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 32, 8>,
                                                    cutlass::gemm::GemmShape<16, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 32, 8], [16, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 32, 8], [16, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_75(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 32, 8>,
                                                    cutlass::gemm::GemmShape<32, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 32, 8], [32, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 32, 8], [32, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_76(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 32, 8>,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 32, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 32, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_77(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 32, 8>,
                                                    cutlass::gemm::GemmShape<96, 4, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 32, 8], [96, 4, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 32, 8], [96, 4, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_78(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 32, 8>,
                                                    cutlass::gemm::GemmShape<96, 8, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 32, 8], [96, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 32, 8], [96, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_79(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 32, 8>,
                                                    cutlass::gemm::GemmShape<96, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 32, 8], [96, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 32, 8], [96, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_80(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 32, 8>,
                                                    cutlass::gemm::GemmShape<96, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 32, 8], [96, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 32, 8], [96, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_81(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 64, 8>,
                                                    cutlass::gemm::GemmShape<16, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 64, 8], [16, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 64, 8], [16, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_82(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 64, 8>,
                                                    cutlass::gemm::GemmShape<32, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 64, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 64, 8], [32, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_83(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 64, 8>,
                                                    cutlass::gemm::GemmShape<48, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 64, 8], [48, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 64, 8], [48, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_84(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 64, 8>,
                                                    cutlass::gemm::GemmShape<96, 8, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 64, 8], [96, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 64, 8], [96, 8, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_85(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 64, 8>,
                                                    cutlass::gemm::GemmShape<96, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 64, 8], [96, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 64, 8], [96, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_86(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 64, 8>,
                                                    cutlass::gemm::GemmShape<96, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 64, 8], [96, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 64, 8], [96, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_87(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 96, 8>,
                                                    cutlass::gemm::GemmShape<16, 96, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 96, 8], [16, 96, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 96, 8], [16, 96, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_88(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 96, 8>,
                                                    cutlass::gemm::GemmShape<32, 96, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 96, 8], [32, 96, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 96, 8], [32, 96, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_89(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 96, 8>,
                                                    cutlass::gemm::GemmShape<96, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 96, 8], [96, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 96, 8], [96, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_90(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 96, 8>,
                                                    cutlass::gemm::GemmShape<96, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 96, 8], [96, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 96, 8], [96, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_91(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 128, 8>,
                                                    cutlass::gemm::GemmShape<48, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 128, 8], [48, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 128, 8], [48, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_92(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 128, 8>,
                                                    cutlass::gemm::GemmShape<96, 16, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 128, 8], [96, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 128, 8], [96, 16, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_93(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 128, 8>,
                                                    cutlass::gemm::GemmShape<96, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 128, 8], [96, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 128, 8], [96, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_94(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 160, 8>,
                                                    cutlass::gemm::GemmShape<96, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 160, 8], [96, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 160, 8], [96, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_95(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 192, 8>,
                                                    cutlass::gemm::GemmShape<16, 192, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 192, 8], [16, 192, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 192, 8], [16, 192, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_96(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 192, 8>,
                                                    cutlass::gemm::GemmShape<32, 96, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 192, 8], [32, 96, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 192, 8], [32, 96, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_97(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 192, 8>,
                                                    cutlass::gemm::GemmShape<48, 64, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 192, 8], [48, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 192, 8], [48, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_98(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 192, 8>,
                                                    cutlass::gemm::GemmShape<96, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 192, 8], [96, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 192, 8], [96, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        


float cutlass_strided_bathed_sgemm_99(
    int m, int n, int k,
    float alpha, float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count, int split_k, int warmup=0
){
    using Gemm = cutlass::gemm::device::GemmBatched<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<96, 224, 8>,
                                                    cutlass::gemm::GemmShape<96, 32, 8>,
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
    
    float total_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i++){
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

        gemm_op.initialize(arguments, workspace.get());
        cutlass::Status status = gemm_op();
        //workspace.release();
        
        if(status != cutlass::Status::kSuccess){
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            
            std::fstream dataFile2;
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[96, 224, 8], [96, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                                    + " ,\"time\": " + std::to_string(-1) + "}";
                
                dataFile2.open(fileName2, std::ios::app);
                
                dataFile2 << json2 << std::endl;
            }

            return -1;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&total_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    
    std::fstream dataFile;
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TT/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[96, 224, 8], [96, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        



int main(int argc, char *argv[]){
    float *rlt = new float[100];

    int M = 64;
    int N = 64;
    int K = 64;
    int Batch = 1;
    int split_k = 1;
    
    int option;
    while((option = getopt(argc, argv, "m:n:k:b:s:")) != -1){
        switch(option){
            case 'm':
                M = std::stoi(optarg);
                break;
            case 'n':
                N = std::stoi(optarg);
                break;
            case 'k':
                K = std::stoi(optarg);
                break;
            case 'b':
                Batch = std::stoi(optarg);
                break;
            case 's':
                split_k = std::stoi(optarg);
            case '?':
                break;
        }
    }
    
    int const lda = K;
    int const ldb = N;
    int const ldc = N;
    
    int const count_A = Batch * M * K;
    int const count_B = Batch * N * K;
    int const count_C = Batch * M * N;
    
    long long int batch_stride_A = static_cast<long long int>(M) * static_cast<long long int>(K);
    long long int batch_stride_B = static_cast<long long int>(K) * static_cast<long long int>(N);
    long long int batch_stride_C = static_cast<long long int>(M) * static_cast<long long int>(N);
    
    float alpha = static_cast<float>(1.0f);
    float beta = static_cast<float>(0.0f);
    
    std::vector<float> host_A(count_A, 1.2f);
    std::vector<float> host_B(count_B, 1.0f);
    std::vector<float> host_C(count_C);
    
    float *A;
    float *B;
    float *C;
    
    cudaMalloc(&A, count_A * sizeof(float));
    cudaMalloc(&B, count_B * sizeof(float));
    cudaMalloc(&C, count_C * sizeof(float));
    
    cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C, host_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice);
    
    //warmp up
    for(int i = 0; i < 20; i++){
        cutlass_strided_bathed_sgemm_0(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, 1, 1);
    }
    
    rlt[0] = cutlass_strided_bathed_sgemm_0(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[1] = cutlass_strided_bathed_sgemm_1(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[2] = cutlass_strided_bathed_sgemm_2(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[3] = cutlass_strided_bathed_sgemm_3(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[4] = cutlass_strided_bathed_sgemm_4(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[5] = cutlass_strided_bathed_sgemm_5(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[6] = cutlass_strided_bathed_sgemm_6(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[7] = cutlass_strided_bathed_sgemm_7(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[8] = cutlass_strided_bathed_sgemm_8(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[9] = cutlass_strided_bathed_sgemm_9(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[10] = cutlass_strided_bathed_sgemm_10(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[11] = cutlass_strided_bathed_sgemm_11(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[12] = cutlass_strided_bathed_sgemm_12(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[13] = cutlass_strided_bathed_sgemm_13(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[14] = cutlass_strided_bathed_sgemm_14(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[15] = cutlass_strided_bathed_sgemm_15(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[16] = cutlass_strided_bathed_sgemm_16(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[17] = cutlass_strided_bathed_sgemm_17(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[18] = cutlass_strided_bathed_sgemm_18(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[19] = cutlass_strided_bathed_sgemm_19(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[20] = cutlass_strided_bathed_sgemm_20(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[21] = cutlass_strided_bathed_sgemm_21(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[22] = cutlass_strided_bathed_sgemm_22(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[23] = cutlass_strided_bathed_sgemm_23(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[24] = cutlass_strided_bathed_sgemm_24(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[25] = cutlass_strided_bathed_sgemm_25(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[26] = cutlass_strided_bathed_sgemm_26(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[27] = cutlass_strided_bathed_sgemm_27(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[28] = cutlass_strided_bathed_sgemm_28(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[29] = cutlass_strided_bathed_sgemm_29(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[30] = cutlass_strided_bathed_sgemm_30(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[31] = cutlass_strided_bathed_sgemm_31(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[32] = cutlass_strided_bathed_sgemm_32(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[33] = cutlass_strided_bathed_sgemm_33(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[34] = cutlass_strided_bathed_sgemm_34(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[35] = cutlass_strided_bathed_sgemm_35(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[36] = cutlass_strided_bathed_sgemm_36(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[37] = cutlass_strided_bathed_sgemm_37(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[38] = cutlass_strided_bathed_sgemm_38(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[39] = cutlass_strided_bathed_sgemm_39(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[40] = cutlass_strided_bathed_sgemm_40(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[41] = cutlass_strided_bathed_sgemm_41(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[42] = cutlass_strided_bathed_sgemm_42(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[43] = cutlass_strided_bathed_sgemm_43(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[44] = cutlass_strided_bathed_sgemm_44(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[45] = cutlass_strided_bathed_sgemm_45(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[46] = cutlass_strided_bathed_sgemm_46(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[47] = cutlass_strided_bathed_sgemm_47(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[48] = cutlass_strided_bathed_sgemm_48(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[49] = cutlass_strided_bathed_sgemm_49(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[50] = cutlass_strided_bathed_sgemm_50(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[51] = cutlass_strided_bathed_sgemm_51(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[52] = cutlass_strided_bathed_sgemm_52(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[53] = cutlass_strided_bathed_sgemm_53(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[54] = cutlass_strided_bathed_sgemm_54(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[55] = cutlass_strided_bathed_sgemm_55(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[56] = cutlass_strided_bathed_sgemm_56(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[57] = cutlass_strided_bathed_sgemm_57(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[58] = cutlass_strided_bathed_sgemm_58(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[59] = cutlass_strided_bathed_sgemm_59(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[60] = cutlass_strided_bathed_sgemm_60(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[61] = cutlass_strided_bathed_sgemm_61(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[62] = cutlass_strided_bathed_sgemm_62(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[63] = cutlass_strided_bathed_sgemm_63(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[64] = cutlass_strided_bathed_sgemm_64(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[65] = cutlass_strided_bathed_sgemm_65(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[66] = cutlass_strided_bathed_sgemm_66(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[67] = cutlass_strided_bathed_sgemm_67(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[68] = cutlass_strided_bathed_sgemm_68(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[69] = cutlass_strided_bathed_sgemm_69(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[70] = cutlass_strided_bathed_sgemm_70(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[71] = cutlass_strided_bathed_sgemm_71(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[72] = cutlass_strided_bathed_sgemm_72(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[73] = cutlass_strided_bathed_sgemm_73(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[74] = cutlass_strided_bathed_sgemm_74(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[75] = cutlass_strided_bathed_sgemm_75(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[76] = cutlass_strided_bathed_sgemm_76(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[77] = cutlass_strided_bathed_sgemm_77(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[78] = cutlass_strided_bathed_sgemm_78(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[79] = cutlass_strided_bathed_sgemm_79(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[80] = cutlass_strided_bathed_sgemm_80(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[81] = cutlass_strided_bathed_sgemm_81(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[82] = cutlass_strided_bathed_sgemm_82(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[83] = cutlass_strided_bathed_sgemm_83(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[84] = cutlass_strided_bathed_sgemm_84(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[85] = cutlass_strided_bathed_sgemm_85(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[86] = cutlass_strided_bathed_sgemm_86(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[87] = cutlass_strided_bathed_sgemm_87(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[88] = cutlass_strided_bathed_sgemm_88(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[89] = cutlass_strided_bathed_sgemm_89(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[90] = cutlass_strided_bathed_sgemm_90(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[91] = cutlass_strided_bathed_sgemm_91(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[92] = cutlass_strided_bathed_sgemm_92(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[93] = cutlass_strided_bathed_sgemm_93(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[94] = cutlass_strided_bathed_sgemm_94(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[95] = cutlass_strided_bathed_sgemm_95(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[96] = cutlass_strided_bathed_sgemm_96(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[97] = cutlass_strided_bathed_sgemm_97(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[98] = cutlass_strided_bathed_sgemm_98(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	rlt[99] = cutlass_strided_bathed_sgemm_99(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, Batch, split_k);

	
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    return 0;
}
        