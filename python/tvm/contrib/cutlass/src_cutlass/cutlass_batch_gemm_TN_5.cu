
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 192, 8>,
                                                    cutlass::gemm::GemmShape<128, 64, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 192, 8], [128, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 192, 8], [128, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 192, 8>,
                                                    cutlass::gemm::GemmShape<128, 96, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 192, 8], [128, 96, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 192, 8], [128, 96, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 192, 8>,
                                                    cutlass::gemm::GemmShape<128, 192, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 192, 8], [128, 192, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 192, 8], [128, 192, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 192, 8>,
                                                    cutlass::gemm::GemmShape<256, 32, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 192, 8], [256, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 192, 8], [256, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 192, 8>,
                                                    cutlass::gemm::GemmShape<256, 48, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 192, 8], [256, 48, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 192, 8], [256, 48, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 192, 8>,
                                                    cutlass::gemm::GemmShape<256, 64, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 192, 8], [256, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 192, 8], [256, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 192, 8>,
                                                    cutlass::gemm::GemmShape<256, 96, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 192, 8], [256, 96, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 192, 8], [256, 96, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 192, 8>,
                                                    cutlass::gemm::GemmShape<256, 192, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 192, 8], [256, 192, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 192, 8], [256, 192, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 224, 8>,
                                                    cutlass::gemm::GemmShape<32, 224, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 224, 8], [32, 224, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 224, 8], [32, 224, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 224, 8>,
                                                    cutlass::gemm::GemmShape<64, 224, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 224, 8], [64, 224, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 224, 8], [64, 224, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 224, 8>,
                                                    cutlass::gemm::GemmShape<128, 112, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 224, 8], [128, 112, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 224, 8], [128, 112, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 224, 8>,
                                                    cutlass::gemm::GemmShape<128, 224, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 224, 8], [128, 224, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 224, 8], [128, 224, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 224, 8>,
                                                    cutlass::gemm::GemmShape<256, 32, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 224, 8], [256, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 224, 8], [256, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 224, 8>,
                                                    cutlass::gemm::GemmShape<256, 112, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 224, 8], [256, 112, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 224, 8], [256, 112, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 224, 8>,
                                                    cutlass::gemm::GemmShape<256, 224, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 224, 8], [256, 224, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 224, 8], [256, 224, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 256, 8>,
                                                    cutlass::gemm::GemmShape<32, 256, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 256, 8], [32, 256, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 256, 8], [32, 256, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 256, 8>,
                                                    cutlass::gemm::GemmShape<64, 128, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 256, 8], [64, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 256, 8], [64, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 256, 8>,
                                                    cutlass::gemm::GemmShape<64, 256, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 256, 8], [64, 256, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 256, 8], [64, 256, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 256, 8>,
                                                    cutlass::gemm::GemmShape<128, 64, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 256, 8], [128, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 256, 8], [128, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 256, 8>,
                                                    cutlass::gemm::GemmShape<128, 128, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 256, 8], [128, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 256, 8], [128, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 256, 8>,
                                                    cutlass::gemm::GemmShape<128, 256, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 256, 8], [128, 256, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 256, 8], [128, 256, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 256, 8>,
                                                    cutlass::gemm::GemmShape<256, 32, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 256, 8], [256, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 256, 8], [256, 32, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 256, 8>,
                                                    cutlass::gemm::GemmShape<256, 64, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 256, 8], [256, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 256, 8], [256, 64, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 256, 8>,
                                                    cutlass::gemm::GemmShape<256, 128, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 256, 8], [256, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 256, 8], [256, 128, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm80,
                                                    cutlass::gemm::GemmShape<256, 256, 8>,
                                                    cutlass::gemm::GemmShape<256, 256, 8>,
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
            std::string fileName2 = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                                    std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
            if(warmup == 0){
                std::string json2 = "{\"dim\": [[256, 256, 8], [256, 256, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
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
    
    std::string fileName = "/home/local_guest/tvm/python/tvm/contrib/cutlass/rlt_cutlass_TN/" + std::to_string(batch_count) + "_" +
                            std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".json";
    if(warmup == 0){
        std::string json = "{\"dim\": [[256, 256, 8], [256, 256, 8], [2], [1]], \"split_k\": " + std::to_string(split_k)
                            + " ,\"time\": " + std::to_string(total_time/20) + "}";
        
        dataFile.open(fileName, std::ios::app);
        
        dataFile << json << std::endl;
    }
    
    return total_time / 20;
    
}
        



int main(int argc, char *argv[]){
    float *rlt = new float[25];

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
    int const ldb = K;
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

	
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    return 0;
}
        