#include <iostream>

int main(void){
    //We will check number of Cuda enebaled devices and their properties
    cudaDeviceProp d_prop;
    int cnt;

    //following fn takes in a pointer & updates the value 
    //CudaDeviceProp is a struct that stores device information
    cudaGetDeviceCount(&cnt);
    printf("cnt of devices is: %d", cnt);
    printf("\n");

    for(int i=0;i<cnt;i++){
        cudaGetDeviceProperties(&d_prop, i);
        printf("the max threds avl are: %d", d_prop.maxThreadsPerBlock);
    }
    printf("\n");
    return 0;
}