#include <iostream>
#include <vector>
#include <iomanip>

#include <cuda_runtime.h>

#define TILE_DIM 16


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

__global__ void copyMatrix2D(int *in, int *out, int width, int height) {    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;
        out[idx] = in[idx];
    }
}

__global__ void transposeMatrix2D(int *in, int *out, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int input_index = row * width + col;
        int output_index = col * height + row;

        out[output_index] = in[input_index];
    }

}

__global__ void transposeMatrixShared2D(int *in, int *out, int width, int height) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    
    __shared__ int shTile[TILE_DIM][TILE_DIM];

    if (col < width && row < height) {
        int idx = row * width + col;
        
        shTile[threadIdx.y][threadIdx.x] = in[idx];
    }

    __syncthreads();  // sync threads in tile

    col = blockIdx.y * TILE_DIM + threadIdx.x; 
    row = blockIdx.x * TILE_DIM + threadIdx.y;


    if (col < height && row < width) {
        int idx = row * height + col;
        
        out[idx] = shTile[threadIdx.x][threadIdx.y];
    }
}



void launchMatrixCopy(int *d_in, int *d_out, int width, int height) {
    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y 
    );

    copyMatrix2D<<<gridSize, blockSize>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();
}

void launchMatrixTranspose(int *d_in, int *d_out, int width, int height) {
    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y 
    );

    transposeMatrix2D<<<gridSize, blockSize>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();
}

void launchMatrixTransposeShared(int *d_in, int *d_out, int width, int height) {
    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y 
    );

    transposeMatrixShared2D<<<gridSize, blockSize>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize(); // sync device
     
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

    // Debug steps, print OG arr
    std::cout << "Transposed Matrix: ";
    printMatrix(h_in, width, height);
    std::cout << std::endl;

    
    int *d_in, *d_out;
    cudaMalloc(&d_in, size_bytes);
    cudaMalloc(&d_out, size_bytes);

    
    cudaMemcpy(d_in, h_in.data(), size_bytes, cudaMemcpyHostToDevice);

    
    launchMatrixTransposeShared(d_in, d_out, width, height);

    
    cudaMemcpy(h_out.data(), d_out, size_bytes, cudaMemcpyDeviceToHost);

    // Debug steps, print Copied arr
    std::cout << "Transposed Matrix: ";
    printMatrix(h_out, height, width);
    std::cout << std::endl;


    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}