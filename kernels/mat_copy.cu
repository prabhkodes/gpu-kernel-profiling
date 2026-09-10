#include <iostream>
#include <vector>
#include <iomanip>

#include <cuda_runtime.h>


__global__ void copyMatrix2D(int *in, int *out, int width, int height) {    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;
        out[idx] = in[idx];
    }
}


void printMatrix(const std::vector<int>& data, int width, int height) {
    std::cout << "\n--- Matrix (" << width << "x" << height << ") ---\n";

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            
            std::cout << std::setw(4) << data[idx] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "-----------------------\n";
}


void launchMatrixCopy(int *d_in, int *d_out, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y 
    );

    copyMatrix2D<<<gridSize, blockSize>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();
}


int main() {
    int width = 10;
    int height = 10;
    int num_elements = width * height;
    size_t size_bytes = num_elements * sizeof(int);


    std::vector<int> h_in(num_elements);
    std::vector<int> h_out(num_elements);


    for (int i = 0; i < num_elements; i++) {
        h_in[i] = i;
    }

    // Debug, print OG arr
    std::cout << "Original Array: ";
    printMatrix(h_in, width, height);
    std::cout << std::endl;

    
    int *d_in, *d_out;
    cudaMalloc(&d_in, size_bytes);
    cudaMalloc(&d_out, size_bytes);

    
    cudaMemcpy(d_in, h_in.data(), size_bytes, cudaMemcpyHostToDevice);

    
    launchMatrixCopy(d_in, d_out, width, height);

    
    cudaMemcpy(h_out.data(), d_out, size_bytes, cudaMemcpyDeviceToHost);

    std::cout << "Copied Array: ";
    printMatrix(h_out, width, height);
    std::cout << std::endl;


    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}