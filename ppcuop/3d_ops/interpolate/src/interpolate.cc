// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate.cpp
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <paddle/extension.h>

#include <vector>

void three_nn_kernel_launcher(int b, int n, int m, const float *unknown,
                              const float *known, float *dist2, int *idx,
                              cudaStream_t stream);

void three_interpolate_kernel_launcher(int b, int c, int m, int n,
                                       const float *points, const int *idx,
                                       const float *weight, float *out,
                                       cudaStream_t stream);

void three_interpolate_grad_kernel_launcher(int b, int c, int n, int m,
                                            const float *grad_out,
                                            const int *idx, const float *weight,
                                            float *grad_points,
                                            cudaStream_t stream);

std::vector<paddle::Tensor> constant_tensor(int constant){
  auto constant_tensor = paddle::Tensor(paddle::PlaceType::kCPU, std::vector<int64_t> {1});
  int* data = constant_tensor.mutable_data<int>(paddle::PlaceType::kCPU);
  data[0] = constant;
  return {constant_tensor};
}

std::vector<paddle::Tensor> three_nn_wrapper(int b, int n, int m, const paddle::Tensor& unknown_tensor,
                                             const paddle::Tensor& known_tensor, const paddle::Tensor& dist2_tensor,
                                             const paddle::Tensor& idx_tensor) {
  const float *unknown = unknown_tensor.data<float>();
  const float *known = known_tensor.data<float>();
  float *dist2 = const_cast<float*>(dist2_tensor.data<float>());
  int *idx = const_cast<int*>(idx_tensor.data<int>());

  cudaStream_t stream = known_tensor.stream();
  three_nn_kernel_launcher(b, n, m, unknown, known, dist2, idx, stream);

  return constant_tensor(1);
}

std::vector<paddle::Tensor> three_interpolate_wrapper(int b, int c, int m, int n,
                                                      const paddle::Tensor& points_tensor, const paddle::Tensor& idx_tensor,
                                                      const paddle::Tensor& weight_tensor,
                                                      const paddle::Tensor& out_tensor) {
  const float *points = points_tensor.data<float>();
  const float *weight = weight_tensor.data<float>();
  float *out = const_cast<float*>(out_tensor.data<float>());
  const int *idx = idx_tensor.data<int>();

  cudaStream_t stream = points_tensor.stream();
  three_interpolate_kernel_launcher(b, c, m, n, points, idx, weight, out,
                                    stream);

  return constant_tensor(1);
}

std::vector<paddle::Tensor> three_interpolate_grad_wrapper(int b, int c, int n, int m,
                                                           const paddle::Tensor& grad_out_tensor,
                                                           const paddle::Tensor& idx_tensor,
                                                           const paddle::Tensor& weight_tensor,
                                                           const paddle::Tensor& grad_points_tensor) {
  const float *grad_out = grad_out_tensor.data<float>();
  const float *weight = weight_tensor.data<float>();
  float *grad_points = const_cast<float*>(grad_points_tensor.data<float>());
  const int *idx = idx_tensor.data<int>();

  cudaStream_t stream = grad_out_tensor.stream();
  three_interpolate_grad_kernel_launcher(b, c, n, m, grad_out, idx, weight,
                                         grad_points, stream);

  return constant_tensor(1);
}

PD_BUILD_OP(three_nn_wrapper)
    .Inputs({"unknown_tensor", "known_tensor", "dist2_tensor", "idx_tensor"})
    .Outputs({"out"})
    .Attrs({"b: int",
            "n: int",
            "m: int"})
    .SetKernelFn(PD_KERNEL(three_nn_wrapper));

PD_BUILD_OP(three_interpolate_wrapper)
    .Inputs({"points_tensor", "idx_tensor", "weight_tensor", "out_tensor"})
    .Outputs({"out"})
    .Attrs({"b: int",
            "c: int",
            "m: int",
            "n: int"})
    .SetKernelFn(PD_KERNEL(three_interpolate_wrapper));

PD_BUILD_OP(three_interpolate_grad_wrapper)
    .Inputs({"grad_out_tensor", "idx_tensor", "weight_tensor", "grad_points_tensor"})
    .Outputs({"out"})
    .Attrs({"b: int",
            "c: int",
            "n: int",
            "m: int"})
    .SetKernelFn(PD_KERNEL(three_interpolate_grad_wrapper));
