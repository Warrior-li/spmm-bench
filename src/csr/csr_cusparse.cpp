#include "matrix.h"

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

//
// The calculation algorithm for the current format
//
double Matrix::calculate() {
    //double start = getTime();
    

    warmup<<<1,1>>>();
    cudaDeviceSynchronize();

    uint64_t *d_rowptr, *d_rowidx;
    float *d_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**)&d_rowptr, (rows+1) * sizeof(uint64_t)) )
    CHECK_CUDA( cudaMalloc((void**)&d_rowidx, coo->nnz * sizeof(uint64_t)) )
    CHECK_CUDA( cudaMalloc((void**)&d_values, coo->nnz * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**)&dB, (rows * cols) * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**)&dC, (rows * cols) * sizeof(float)) )
    
    CHECK_CUDA( cudaMemcpy(d_rowptr, rowptr, (rows+1) * sizeof(uint64_t), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_rowidx, rowidx, coo->nnz * sizeof(uint64_t), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_values, values, coo->nnz * sizeof(float), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, B, (rows * cols) * sizeof(float), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, C, (rows * cols) * sizeof(float), cudaMemcpyHostToDevice) )
    
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    float  alpha       = 1.0f;
    float  beta        = 0.0f;
    
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, rows, cols, coo->nnz,
                          d_rowptr, d_rowidx, d_values,
                          CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    
    // Create dense matrix B
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, rows, cols, rows, dB,
                            CUDA_R_32F, CUSPARSE_ORDER_ROW))
    
    // Create dense matrix C
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, cols, rows, cols, dC,
                            CUDA_R_32F, CUSPARSE_ORDER_ROW))
    
    // Allocate external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(handle,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                             CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    

    // ---------- 3. 用 CUDA Events 计时 ----------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 确保之前所有操作完成后再开始计时
    cudaDeviceSynchronize();

    // 开始计时（记录 GPU 时间戳）
    cudaEventRecord(start, 0);

    // Execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 CUSPARSE_OPERATION_TRANSPOSE,
                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

    // 结束计时
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double kernel_time = milliseconds / 1000.0; // 转成秒

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Destroy matrix descriptors
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);
    
    // Copy C back
    cudaMemcpy(C, dC, (rows * cols) * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free all the memory
    cudaFree(dBuffer);
    cudaFree(d_rowptr);
    cudaFree(d_rowidx);
    cudaFree(d_values);
    cudaFree(dB);
    cudaFree(dC);
    
    //double end = getTime();
    return kernel_time;
}

