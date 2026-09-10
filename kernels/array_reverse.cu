#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>


__global__ void reverse_arr(int * data, int n) { 
    int temp;
    int last_idx;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n/2 )
    {
            last_idx = n - 1 - i ;
            temp = data[i];

            data[i] = data[ last_idx];
            data[last_idx] = temp;
    }
}


void reverseHostArray(std::vector<int> &data){

    int * device_data;
    int n = data.size();
    
    size_t size_in_bytes =  n * sizeof(int);
    cudaMallocManaged(&device_data, size_in_bytes); // cuda alloc

    cudaMemcpy(device_data, data.data(), size_in_bytes, cudaMemcpyHostToDevice);


    int nthreads = 2048*2048;
    int block_size = 256; 
    int grid_size =  (int)ceil(nthreads/block_size);
    reverse_arr <<<grid_size, block_size>>> (device_data, n); // define kernel
        
    cudaDeviceSynchronize(); 
    
    
    cudaMemcpy(data.data(), device_data, size_in_bytes, cudaMemcpyDeviceToHost);
    
    
    cudaFree(device_data);

}


int main() {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Debug steps, print OG arr
    std::cout << "Original Array: ";
    for (int x : data) std::cout << x << " ";
    std::cout << std::endl;
    
    
    // reverse array
    reverseHostArray(data);

    // Debug Steps, print rev arr
    std::cout << "Reversed Array: ";
    for (int x : data) std::cout << x << " ";
    std::cout << std::endl;

}
