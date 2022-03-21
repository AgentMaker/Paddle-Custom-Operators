#include <paddle/extension.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <vector>

void gather_points_kernel_launcher(int b, int c, int n, int npoints,
                                   const float *points, const int *idx,
                                   float *out, cudaStream_t stream);

void gather_points_grad_kernel_launcher(int b, int c, int n, int npoints,
                                        const float *grad_out, const int *idx,
                                        float *grad_points,
                                        cudaStream_t stream);

std::vector<paddle::Tensor> constant_tensor(int constant){
  auto constant_tensor = paddle::Tensor(paddle::PlaceType::kCPU, std::vector<int64_t> {1});
  int* data = constant_tensor.mutable_data<int>(paddle::PlaceType::kCPU);
  data[0] = constant;
  return {constant_tensor};
}

std::vector<paddle::Tensor> gather_points_wrapper(int b, int c, int n, int npoints,
                                                  const paddle::Tensor& points_tensor, const paddle::Tensor& idx_tensor,
                                                  const paddle::Tensor& out_tensor) {
  const float *points = points_tensor.data<float>();
  const int *idx = idx_tensor.data<int>();
  float *out = const_cast<float*>(out_tensor.data<float>());

  cudaStream_t stream = points_tensor.stream();
  gather_points_kernel_launcher(b, c, n, npoints, points, idx, out, stream);
  
  return constant_tensor(1);
}

std::vector<paddle::Tensor> gather_points_grad_wrapper(int b, int c, int n, int npoints,
                                                       const paddle::Tensor& grad_out_tensor,
                                                       const paddle::Tensor& idx_tensor,
                                                       const paddle::Tensor& grad_points_tensor) {
  const float *grad_out = grad_out_tensor.data<float>();
  const int *idx = idx_tensor.data<int>();
  float *grad_points = const_cast<float*>(grad_points_tensor.data<float>());

  cudaStream_t stream = grad_out_tensor.stream();
  gather_points_grad_kernel_launcher(b, c, n, npoints, grad_out, idx,
                                     grad_points, stream);
  
  return constant_tensor(1);
}

PD_BUILD_OP(gather_points_wrapper)
    .Inputs({"points_tensor", "idx_tensorr", "out_tensor"})
    .Outputs({"out"})
    .Attrs({"b: int",
            "c: int",
            "n: int",
            "npoints: int"})
    .SetKernelFn(PD_KERNEL(gather_points_wrapper));

PD_BUILD_OP(gather_points_grad_wrapper)
    .Inputs({"grad_out_tensor", "idx_tensor", "grad_points_tensor"})
    .Outputs({"out"})
    .Attrs({"b: int",
            "c: int",
            "n: int",
            "npoints: int"})
    .SetKernelFn(PD_KERNEL(gather_points_grad_wrapper));