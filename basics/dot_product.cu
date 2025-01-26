#include <iostream>
#define N 1000

int imin(int a, int b){
    return a<b? a:b;
}


//reduction will require decent number of blocks which we arbitrarily set to 16 here.
//however if the #elems are too few we would only require blocks sufficient to
// carry out the summation

const int block_size = 256; //num of threads per block
const int grid_size = imin(16, (N + block_size-1)/N); //num of blocks per grid

__global__ void naive_dot_product(int *d_vec1, int *d_vec2, int *c, int n){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int cache_arr[block_size];
    int temp = 0;
    int cache_idx = threadIdx.x;

    while(tid < n){
        temp += d_vec1[tid] * d_vec2[tid];
        int offset = blockDim.x * gridDim.x;
        tid += offset;
    }

    cache_arr[cache_idx] = temp;
    __syncthreads();

    int cnt = blockDim.x/2;
    while(cnt != 0){
        if(threadIdx.x < cnt)
         cache_arr[threadIdx.x] += cache_arr[threadIdx.x + cnt];
        __syncthreads();
        cnt /= 2;
    }

    __syncthreads();

    if(threadIdx.x == 0){
        c[blockIdx.x] = cache_arr[0];
    }

}

int main(){
    int *h_vec1, *h_vec2, *h_out;
    int *d_vec1, *d_vec2, *d_out;

    int ans = 0;

    h_vec1 = (int*)malloc(N* sizeof(int));
    h_vec2 = (int*)malloc(N* sizeof(int));
    h_out =  (int*)malloc(grid_size* sizeof(int));
    
    for(int i = 0; i < N; i++){
        h_vec1[i] = i;
        h_vec2[i] = 2*i;
    }



    //allocate memory

    cudaMalloc((void**) &d_vec1, N* sizeof(int));
    cudaMalloc((void**) &d_vec2, N* sizeof(int));
    cudaMalloc((void**) &d_out,  grid_size* sizeof(int));

    //transfer data from host to device

    cudaMemcpy(d_vec1, h_vec1, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2, N*sizeof(int), cudaMemcpyHostToDevice);

    //launch kernel
    naive_dot_product<<<grid_size, block_size>>>(d_vec1, d_vec2, d_out, N);

    //copy results from device to host
    cudaMemcpy(h_out, d_out, grid_size*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i =0;i<grid_size;i++){
        ans+=h_out[i];
    }
    
    int ANS = ((N)*(N+1)*(2*N+1))/3;
    printf("Ans should be: %d \n", ANS);
    printf("Ans is: %d \n", ans);

   
    return 0;
}