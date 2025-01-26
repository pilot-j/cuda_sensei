#include <iostream>
#define N 10

//kernel to add vectors larger than the actual number of threads * blocks
__global__ void vec_add(int *d_mat1, int *d_mat2, int *d_out, int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < n){
        d_out[tid] = d_mat1[tid] + d_mat2[tid];
        int offset = blockDim.x * gridDim.x;
        tid += offset;
    }
}

/*
//basic kernel with parallelisation across threads
__global__ void vec_add(int *d_v1, int *d_v2, int *d_out, int n ) {
    int tid = threadIdx.x;

    if(tid<n){
        d_out[tid]= d_v1[tid] + d_v2[tid];
    }
}
*/

int main(void){
    int h_vec1[N], h_vec2[N], h_out[N];
    int *d_vec1, *d_vec2, *d_out;
    
    dim3 grid_size(1); // we take 1 blocks with N threads
    dim3 block_size(N);

    /*Alternate way to launch kernels
    int threads = x
    dim3 block_size(x);
    dim3 grid_size((N + threads - 1) / threads); 
    */

    for(int i =0;i<N;i++){
        h_vec1[i]= 2*i;
        h_vec2[i]= i;
    }
    //in usual cases we will receive these host arrays from somewhere

    cudaMalloc((void**)&d_vec1, N*sizeof(int));
    cudaMalloc((void**)&d_vec2, N*sizeof(int));
    cudaMalloc((void**)&d_out,  N*sizeof(int));

    cudaMemcpy(d_vec1, h_vec1,N*sizeof(int) ,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2,N*sizeof(int) ,cudaMemcpyHostToDevice);
    
    vec_add<<<grid_size, block_size>>>(d_vec1, d_vec2, d_out, N);

    cudaMemcpy(h_out,  d_out ,N*sizeof(int) ,cudaMemcpyHostToDevice);

    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_out);
     
    for(int i =0;i<N;i++){
        printf("%d + %d = %d \n", h_vec1[i], h_vec2[i], h_out[i]);
    }

    return 0;
}
