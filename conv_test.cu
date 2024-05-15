#include <cuda.h>
#include <stdio.h>
#include <torch/torch.h>
#include <iostream>

namespace F = torch::nn::functional;

#define WINDOW_SIZE 11

__constant__ float window[WINDOW_SIZE * WINDOW_SIZE]; // 11x11 gaussian filter

__global__ void tiledConvCUDA(
    float *gt,
    float *input,
    int C,
    int H,
    int W,
    float *output)
{
  extern __shared__ float shared[]; // size: TILE_SIZE - WINDOW_SIZE + 1

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z;
  int index = c * W * H + y * W + x;

  const int TILE_SIZE = 16;
  const int SM_SIZE = TILE_SIZE + WINDOW_SIZE - 1; // Shared memory size.
  int row = threadIdx.x;
  int col = threadIdx.y;
  int tx = blockIdx.x;
  int ty = blockIdx.y;

  for (int j = 0; j < SM_SIZE; j += TILE_SIZE)
  {
    for (int i = 0; i < SM_SIZE; i += TILE_SIZE)
    {
      // Load data to shared[] rec: [(i, j), (i+K, j+K)].
      int target_x = tx * TILE_SIZE - WINDOW_SIZE / 2 + i + row;
      int target_y = ty * TILE_SIZE - WINDOW_SIZE / 2 + j + col;
      printf("i=%d, j=%d, target_x=%d, target_y=%d\n", i, j, target_x, target_y);

      if ((target_x < (W + WINDOW_SIZE / 2)) && (target_y < (H + WINDOW_SIZE / 2)))
      {
        if (target_x < 0 || target_x > W || target_y < 0 || target_y > H)
          shared[(i + row) * SM_SIZE + (j + col)] = 0;
        else
        {
          printf("--sm(%d, %d)--:%d\n", i + row, j + col, c * W * H + target_x * W + target_y);
          shared[(i + row) * SM_SIZE + (j + col)] = input[c * W * H + target_x * W + target_y];
        }
        
        printf("Load to sm(%d, %d): %f\n", i + row, j + col, shared[(i + row) * SM_SIZE + (j + col)]);
      }
    }
  }
  __syncthreads();

  float sum = 0;
  for (int i = 0; i < WINDOW_SIZE; i++)
  {
    for (int j = 0; j < WINDOW_SIZE; j++)
    {
      sum += shared[(row + i) * SM_SIZE + (col + j)] * window[i * WINDOW_SIZE + j];
    }
  }

  output[index] = sum;
}

int main()
{
  // auto options = torch::TensorOptions().device(torch::kCUDA);
  torch::Tensor x = torch::randn({1, 3, 1000, 1000}).cuda().requires_grad_(true);
  torch::Tensor filter = torch::randn({11, 11}).cuda();

  torch::Tensor y = F::conv2d(x, filter.unsqueeze(0).unsqueeze(0).expand({3, 1, 11, 11}), F::Conv2dFuncOptions().padding(1).stride(1).groups(3));
  torch::Tensor loss = torch::sum(y);
  loss.backward();
  // std::cout << "y=" << y << std::endl;
  // std::cout << "grad_x=" << x.grad() << std::endl;

  torch::Tensor z = torch::zeros({1, 3, 1000, 1000}).cuda();
  int C = x.size(1);
  int H = x.size(2);
  int W = x.size(3);

  float *filter_ptr = filter.contiguous().data<float>();
  cudaMemcpyToSymbol(window, filter_ptr, sizeof(filter_ptr));

  const int BLOCK_SIZE = 16;
  const int SM_SIZE = BLOCK_SIZE + WINDOW_SIZE - 1;
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((W + dimBlock.x - 1) / dimBlock.x, (H + dimBlock.y - 1) / dimBlock.y, C);

  tiledConvCUDA<<<dimGrid, dimBlock, SM_SIZE * SM_SIZE * sizeof(float)>>>(
    x.contiguous().data<float>(),
    C,
    H,
    W,
    z.contiguous().data<float>());
  
  cudaDeviceSynchronize();
  return 0;
}