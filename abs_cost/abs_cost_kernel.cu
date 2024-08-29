#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#define BLOCK 16

// (B,H,W1,C) (B,H,W2,C) -> (B,H,W1,W2)

__forceinline__ __device__ bool within_bounds(int h, int w1, int w2, int H, int W1, int W2) {
  return h >= 0 && h < H && w1 >= 0 && w1 < W1 && w2 >= 0 && w2 < W2;
}

template <typename scalar_t>
__global__ void absolute_difference_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap2,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> result)
{
  const int C  = fmap1.size(3);
  const int H  = fmap1.size(1);
  const int W1 = fmap1.size(2);
  const int W2 = fmap2.size(2);

  // 获取当前线程的索引
  const int w1 = blockIdx.x * blockDim.x + threadIdx.x;
  const int w2 = blockIdx.y * blockDim.y + threadIdx.y;
  const int h  = blockIdx.z % H;
  const int b  = blockIdx.z / H;

  if (!within_bounds(h, w1, w2, H, W1, W2)) {
    return;
  }

  scalar_t sum = 0.0;
  for (int c = 0; i < C; ++c) {
    scalar_t diff = fabs(fmap1[b][h][w1][c] - fmap2[b][h][w2][c]);
    sum += diff;
  }

  result[b][h][w1][w2] = sum;
}

template <typename scalar_t>
__global__ void absolute_difference_backward_kernel_fmap1(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap2,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_output,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_fmap1)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int h = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z;

  const int i_size = fmap1.size(1);
  const int j_size = fmap1.size(2);
  const int k_size = fmap1.size(3);
  const int h_size = fmap2.size(3);

  if (!within_bounds(h, k, j_size, k_size)) {
    return;
  }

  for (int i = 0; i < i_size; ++i) {
    for (int j = 0; j < j_size; ++j) {
      scalar_t grad = 0.0;

      scalar_t diff = fmap1[n][i][j][k] - fmap2[n][i][j][h];
      if (diff >= 0) {
        grad = grad_output[n][h][k][h];
      } else {
        grad = -grad_output[n][h][k][h];
      }

      grad_fmap1[n][i][j][k] += grad;
    }
  }
}

template <typename scalar_t>
__global__ void absolute_difference_backward_kernel_fmap2(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap2,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_output,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_fmap2)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int h = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z;

  const int i_size = fmap1.size(1);
  const int j_size = fmap1.size(2);
  const int k_size = fmap1.size(3);
  const int h_size = fmap2.size(3);

  if (!within_bounds(h, k, j_size, k_size)) {
    return;
  }

  for (int i = 0; i < i_size; ++i) {
    for (int j = 0; j < j_size; ++j) {
      scalar_t grad = 0.0;

      scalar_t diff = fmap2[n][i][j][h] - fmap1[n][i][j][k];
      if (diff >= 0) {
        grad = grad_output[n][h][k][h];
      } else {
        grad = -grad_output[n][h][k][h];
      }

      grad_fmap2[n][i][j][h] += grad;
    }
  }
}

/**
 * compute correlation between each element (h,w1)~(h,w2).
 * (B,H,W1,C) (B,H,W2,C) -> (B,H,W1,W2)
 */
std::vector<torch::Tensor> absolute_difference_cuda_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2)
{
  const auto B  = fmap1.size(0);
  const auto H  = fmap1.size(1);
  const auto W1 = fmap1.size(2);
  const auto W2 = fmap2.size(2);

  const dim3 blocks((W1 + BLOCK - 1) / BLOCK, 
                    (W2 + BLOCK - 1) / BLOCK,
                    B*H);

  const dim3 threads(BLOCK, BLOCK);

  auto opts = fmap1.options();
  torch::Tensor result = torch::zeros({B, H, W1, W2}, opts);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.scalar_type(), "absolute_difference_forward_kernel", ([&] {
    absolute_difference_forward_kernel<scalar_t><<<blocks, threads>>>(
      fmap1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      fmap2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      result.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));

  return {result};
}

std::vector<torch::Tensor> absolute_difference_cuda_backward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor grad_output)
{
  const auto B  = fmap1.size(0);
  const auto H  = fmap1.size(1);
  const auto W1 = fmap1.size(2);
  const auto W2 = fmap2.size(2);

  auto grad_fmap1 = torch::zeros_like(fmap1);
  auto grad_fmap2 = torch::zeros_like(fmap2);

  const dim3 blocks((k_size + BLOCK - 1) / BLOCK,
                    (h_size + BLOCK - 1) / BLOCK,
                    batch_size);

  const dim3 threads(BLOCK, BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.scalar_type(), "absolute_difference_backward_kernel_fmap1", ([&] {
    absolute_difference_backward_kernel_fmap1<scalar_t><<<blocks, threads>>>(
      fmap1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      fmap2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      grad_fmap1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap2.scalar_type(), "absolute_difference_backward_kernel_fmap2", ([&] {
    absolute_difference_backward_kernel_fmap2<scalar_t><<<blocks, threads>>>(
      fmap1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      fmap2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      grad_fmap2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));

  return {grad_fmap1, grad_fmap2};
}
