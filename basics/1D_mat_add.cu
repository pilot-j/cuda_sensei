#include <iostream>
#include <cuda_runtime.h>


#define n_row 100
#define n_col 100



__global__ void mat_add(float *d_matA, float *d_matB, float *d_out, int N){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //if num of threads is greater than tot_elem some threads will be idle
    //can we optimise??
    
    while(tid < N){
        d_out[tid] = d_matA[tid] + d_matB[tid];

        //if block_size < tot_elem
        tid += blockDim.x * gridDim.x;
    }

}

int main(){
    int tot_elem = n_row * n_col;
    int mem_size = tot_elem * sizeof(float);

    //create 1D equivalent of a 2D matrix
    float *h_1D_matA, *h_1D_matB, *h_out;
    h_1D_matA = (float*)malloc(mem_size);
    h_1D_matB = (float*)malloc(mem_size);
    h_out     = (float*)malloc(mem_size);

    for(int i=0;i<n_row;i++){
        for(int j =0;j<n_col;j++){
            h_1D_matA[i*n_col + j] = i+j;
            h_1D_matB[i*n_col + j] = i;
        }
    }

    //allocate device memory
    float *d_matA, *d_matB, *d_out;
    cudaMalloc((void**)&d_matA, mem_size);
    cudaMalloc((void**)&d_matB, mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    //copy contents to device
    cudaMemcpy(d_matA, h_1D_matA, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, h_1D_matB, mem_size, cudaMemcpyHostToDevice);

    //launch kernel
    int block_size = 256; // #threads per block.
    int grid_size = (tot_elem + block_size - 1) / block_size; //#blocks per grid

    mat_add<<<grid_size, block_size>>>(d_matA, d_matB, d_out, tot_elem);

    //copy answer back to host
    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
            std::cout << h_out[i * n_col + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    free(h_1D_matA);
    free(h_1D_matB);
    free(h_out);
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_out);

    return 0;
}