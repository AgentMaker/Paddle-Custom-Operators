// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/sampling.cpp

#include <paddle/extension.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <vector>

void furthest_point_sampling_kernel_launcher(int b, int n, int m,
                                             const float *dataset, float *temp,
                                             int *idxs, cudaStream_t stream);

void furthest_point_sampling_with_dist_kernel_launcher(int b, int n, int m,
                                                       const float *dataset,
                                                       float *temp, int *idxs,
                                                       cudaStream_t stream);

std::vector<paddle::Tensor> constant_tensor(int constant){
  auto constant_tensor = paddle::Tensor(paddle::PlaceType::kCPU, std::vector<int64_t> {1});
  int* data = constant_tensor.mutable_data<int>(paddle::PlaceType::kCPU);
  data[0] = constant;
  return {constant_tensor};
}

std::vector<paddle::Tensor> furthest_point_sampling_wrapper(int b, int n, int m,
                                                            const paddle::Tensor& points_tensor,
                                                            const paddle::Tensor& temp_tensor,
                                                            const paddle::Tensor& idx_tensor) {
  const float *points = points_tensor.data<float>();
  float *temp = const_cast<float*>(temp_tensor.data<float>());
  int *idx = const_cast<int*>(idx_tensor.data<int>());

  cudaStream_t stream = points_tensor.stream();
  furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx, stream);
  
  return constant_tensor(1);
}

std::vector<paddle::Tensor> furthest_point_sampling_with_dist_wrapper(int b, int n, int m,
                                                                      const paddle::Tensor& points_tensor,
                                                                      const paddle::Tensor& temp_tensor,
                                                                      const paddle::Tensor& idx_tensor) {

  const float *points = points_tensor.data<float>();
  float *temp = const_cast<float*>(temp_tensor.data<float>());
  int *idx = const_cast<int*>(idx_tensor.data<int>());

  cudaStream_t stream = points_tensor.stream();
  furthest_point_sampling_with_dist_kernel_launcher(b, n, m, points, temp, idx, stream);
  
  return constant_tensor(1);
}

PD_BUILD_OP(furthest_point_sampling_wrapper)
    .Inputs({"points_tensor", "temp_tensor", "idx_tensor"})
    .Outputs({"out"})
    .Attrs({"b: int",
            "n: int",
            "m: int"})
    .SetKernelFn(PD_KERNEL(furthest_point_sampling_wrapper));

PD_BUILD_OP(furthest_point_sampling_with_dist_wrapper)
    .Inputs({"points_tensor", "temp_tensor", "idx_tensor"})
    .Outputs({"out"})
    .Attrs({"b: int",
            "n: int",
            "m: int"})
    .SetKernelFn(PD_KERNEL(furthest_point_sampling_with_dist_wrapper));