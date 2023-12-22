#include <stdio.h>
#include <cuda_runtime.h>

__global__ void myKernel() {
    // 只让 blockIdx.x == 0 的线程执行 printf
    // if (blockIdx.x == 0 && threadIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0) {
    //     printf("Hello from blockIdx.x = %d, threadIdx.x = %d\n", blockIdx.x, threadIdx.x);
    // }

    // 这里可以添加其他核函数的计算逻辑

    // block index
    int bx = blockIdx.x;  // 0 ~ 1023
    int by = blockIdx.y;  // 0 ~ 10
    printf("by:%d \n", by);

    // thread index
    int tx = threadIdx.x; // 0 ~ 31
    int ty = threadIdx.y; // 0 ~ 31
    // printf("ty:%d \n", ty);

    int global_tid_x = bx * blockDim.x + tx; // 0 ~ N - 1
    int global_tid_y = by * blockDim.y + ty; // 0 ~ gridDim.y - 1
    // printf("global_tid_x:%d \n", global_tid_x);
    printf("global_tid_y:%d \n", global_tid_y);
}

int main() {
    // 设置 GPU 设备
    cudaSetDevice(0);

    // 定义 grid 和 block 的大小
    dim3 gridSize(5, 11);  // 2 个块
    dim3 blockSize(4, 32);  // 每个块 64 个线程

    // 调用核函数
    myKernel<<<gridSize, blockSize>>>();

    // 等待 GPU 完成所有任务
    cudaDeviceSynchronize();

    return 0;
}
