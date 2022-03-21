// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query.cpp

#include <paddle/extension.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

void ball_query_kernel_launcher(int b, int n, int m, float min_radius, float max_radius,
                                int nsample, const float *xyz, const float *new_xyz,
                                int *idx, cudaStream_t stream);

std::vector<paddle::Tensor> constant_tensor(int constant){
  auto constant_tensor = paddle::Tensor(paddle::PlaceType::kCPU, std::vector<int64_t> {1});
  int* data = constant_tensor.mutable_data<int>(paddle::PlaceType::kCPU);
  data[0] = constant;
  return {constant_tensor};
}

std::vector<paddle::Tensor> ball_query_wrapper(int b, int n, int m, float min_radius, float max_radius, int nsample,
                                                const paddle::Tensor &new_xyz_tensor, const paddle::Tensor &xyz_tensor,
                                                const paddle::Tensor &idx_tensor) {
  
  CHECK_INPUT(new_xyz_tensor);
  CHECK_INPUT(xyz_tensor);

  const float *new_xyz = new_xyz_tensor.data<float>();
  const float *xyz = xyz_tensor.data<float>();
  int *idx = const_cast<int*>(idx_tensor.data<int>());

  cudaStream_t stream = xyz_tensor.stream();
  ball_query_kernel_launcher(b, n, m, min_radius, max_radius,
                             nsample, new_xyz, xyz, idx, stream);

  return constant_tensor(1);
}

PD_BUILD_OP(ball_query_wrapper)
    .Inputs({"new_xyz_tensor", "xyz_tensor", "idx_tensor"})
    .Outputs({"out"})
    .Attrs({"b: int",
            "n: int",
            "m: int",
            "min_radius: float",
            "max_radius: float",
            "nsample: int"})
    .SetKernelFn(PD_KERNEL(ball_query_wrapper));