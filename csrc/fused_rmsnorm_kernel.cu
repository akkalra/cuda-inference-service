// Fused RMSNorm CUDA kernel: y = x * rsqrt(mean(x^2) + eps) * weight
// One thread block per row, warp-shuffle reduction.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#define WARP_SIZE 32

template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    val += __shfl_xor_sync(0xffffffff, val, offset);
  return val;
}

template <typename T>
__inline__ __device__ T block_reduce_sum(T val) {
  static __shared__ T shared[WARP_SIZE];
  const int lane = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;
  val = warp_reduce_sum(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : (T)0.0f;
  if (wid == 0) val = warp_reduce_sum(val);
  return val;
}

template <typename scalar_t>
__global__ void fused_rmsnorm_kernel(scalar_t* __restrict__ out,
                                     const scalar_t* __restrict__ input,
                                     const scalar_t* __restrict__ weight,
                                     const float epsilon,
                                     const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  const int row = blockIdx.x;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[row * hidden_size + idx];
    variance += x * x;
  }
  variance = block_reduce_sum<float>(variance);
  if (threadIdx.x == 0) s_variance = rsqrtf(variance / hidden_size + epsilon);
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[row * hidden_size + idx];
    out[row * hidden_size + idx] = ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

void fused_rmsnorm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      input.scalar_type(), "fused_rmsnorm_kernel", [&] {
        fused_rmsnorm_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(), (float)epsilon, hidden_size);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
