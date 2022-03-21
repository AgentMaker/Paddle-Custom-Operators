// Modified from https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/pointops/src/knnquery_heap
#include <paddle/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

void knn_kernel_launcher(
    int b,
    int n,
    int m,
    int nsample,
    const float *xyz,
    const float *new_xyz,
    int *idx,
    float *dist2,
    cudaStream_t stream
    );

std::vector<paddle::Tensor> constant_tensor(int constant){
  auto constant_tensor = paddle::Tensor(paddle::PlaceType::kCPU, std::vector<int64_t> {1});
  int* data = constant_tensor.mutable_data<int>(paddle::PlaceType::kCPU);
  data[0] = constant;
  return {constant_tensor};
}

std::vector<paddle::Tensor> knn_wrapper(int b, int n, int m, int nsample, 
                                        const paddle::Tensor& xyz_tensor, const paddle::Tensor& new_xyz_tensor, 
                                        const paddle::Tensor& idx_tensor, const paddle::Tensor& dist2_tensor){
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);

    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    int *idx = const_cast<int*>(idx_tensor.data<int>());
    float *dist2 = const_cast<float*>(dist2_tensor.data<float>());

    cudaStream_t stream = xyz_tensor.stream();

    knn_kernel_launcher(b, n, m, nsample, xyz, new_xyz, idx, dist2, stream);

    return constant_tensor(1);
}


PD_BUILD_OP(knn_wrapper)
    .Inputs({"xyz_tensor", "new_xyz_tensor", "idx_tensor", "dist2_tensor"})
    .Outputs({"out"})
    .Attrs({"b: int",
            "n: int",
            "m: int",
            "nsample: int"})
    .SetKernelFn(PD_KERNEL(knn_wrapper));
