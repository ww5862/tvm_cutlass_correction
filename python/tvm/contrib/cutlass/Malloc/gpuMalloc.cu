//Maloc gpu memory and Memcpy host memory to device memory

#include<iostream>
#include<cuda_runtime.h>
#include<vector>
#include "data.h"

#define EXPORT



extern "C"{
    EXPORT void *MallocFloat(float *a, int batch, int row, int column, int size){
        Data *data = new Data();

        data->batch = batch;
        data->row = row;
        data->column = column;

        void *&p = data->p;

        cudaMalloc(&p, size * sizeof(float));
        cudaMemcpy(p, a, size * sizeof(float), cudaMemcpyHostToDevice);

        return data;
    }

    EXPORT void freeMalloc(void *a){
        Data *data = reinterpret_cast<Data *>(a);
        cudaFree(data->p);
        free(data);
    }

    EXPORT void memsetZero(void *a, int len){
        Data *data = reinterpret_cast<Data *>(a);
        float *A = reinterpret_cast<float *>(data->p);

        cudaMemset(A, 0, len * sizeof(float));
    }

    EXPORT void verify(void *d_a, int size){
        Data *data = reinterpret_cast<Data *>(d_a);
        std::vector<float> h_a(size);

        cudaMemcpy(h_a.data(), data->p, size * sizeof(float), cudaMemcpyDeviceToHost);

        for(auto &idx: h_a) std::cout << idx << " ";
        std::cout << std::endl;
    }

    EXPORT void resetGPU(){
        cudaDeviceReset();
        cudaDeviceSynchronize();
    }
}