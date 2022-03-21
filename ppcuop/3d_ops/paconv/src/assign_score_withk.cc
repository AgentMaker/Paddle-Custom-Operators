// Modified from https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/paconv_lib/src/gpu
#include <paddle/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

void assign_score_withk_forward_launcher(int B, int N0, int N1, int M, int K, int O, int aggregate,
                                        const float* points_data,
                                        const float* centers_data,
                                        const float* scores_data,
                                        const int64_t* knn_idx_data,
                                        float* output_data, 
                                        cudaStream_t stream);

void assign_score_withk_backward_launcher(int B, int N0, int N1, int M, int K, int O, int aggregate,
                                         const float* grad_out_data,
                                         const float* points_data,
                                         const float* centers_data,
                                         const float* scores_data,
                                         const int64_t* knn_idx_data,
                                         float* grad_points_data,
                                         float* grad_centers_data,
                                         float* grad_scores_data,
                                         cudaStream_t stream);

std::vector<paddle::Tensor> constant_tensor(int constant){
  auto constant_tensor = paddle::Tensor(paddle::PlaceType::kCPU, std::vector<int64_t> {1});
  int* data = constant_tensor.mutable_data<int>(paddle::PlaceType::kCPU);
  data[0] = constant;
  return {constant_tensor};
}

std::vector<paddle::Tensor> assign_score_withk_forward_wrapper(int B, int N0, int N1, int M, int K, int O, int aggregate,
                                                                const paddle::Tensor& points,
                                                                const paddle::Tensor& centers,
                                                                const paddle::Tensor& scores,
                                                                const paddle::Tensor& knn_idx,
                                                                const paddle::Tensor& output) {
    CHECK_INPUT(points);
    CHECK_INPUT(centers);
    CHECK_INPUT(scores);
    CHECK_INPUT(knn_idx);
    CHECK_INPUT(output);

    const float* points_data = points.data<float>();
    const float* centers_data = centers.data<float>();
    const float* scores_data = scores.data<float>();
    const int64_t* knn_idx_data = knn_idx.data<int64_t>();
    float* output_data = const_cast<float*>(output.data<float>());
    cudaStream_t stream = points.stream();

    assign_score_withk_forward_launcher(B, N0, N1, M, K, O, aggregate, points_data, centers_data, scores_data, knn_idx_data, output_data, stream);

    return constant_tensor(1);
}

std::vector<paddle::Tensor> assign_score_withk_backward_wrapper(int B, int N0, int N1, int M, int K, int O, int aggregate,
                                                                const paddle::Tensor& grad_out,
                                                                const paddle::Tensor& points,
                                                                const paddle::Tensor& centers,
                                                                const paddle::Tensor& scores,
                                                                const paddle::Tensor& knn_idx,
                                                                const paddle::Tensor& grad_points,
                                                                const paddle::Tensor& grad_centers,
                                                                const paddle::Tensor& grad_scores) {

    CHECK_INPUT(grad_out);
    CHECK_INPUT(scores);
    CHECK_INPUT(points);
    CHECK_INPUT(centers);
    CHECK_INPUT(knn_idx);
    CHECK_INPUT(grad_scores);
    CHECK_INPUT(grad_points);
    CHECK_INPUT(grad_centers);

    const float* grad_out_data = grad_out.data<float>();
    const float* points_data = points.data<float>();
    const float* centers_data = centers.data<float>();
    const float* scores_data = scores.data<float>();
    const int64_t* knn_idx_data = knn_idx.data<int64_t>();
    float* grad_points_data = const_cast<float*>(grad_points.data<float>());
    float* grad_centers_data = const_cast<float*>(grad_centers.data<float>());
    float* grad_scores_data = const_cast<float*>(grad_scores.data<float>());
    cudaStream_t stream = grad_out.stream();

    assign_score_withk_backward_launcher(B, N0, N1, M, K, O, aggregate, grad_out_data, points_data, centers_data, scores_data, knn_idx_data, grad_points_data, grad_centers_data, grad_scores_data, stream);

    return constant_tensor(1);
}

PD_BUILD_OP(assign_score_withk_forward_wrapper)
    .Inputs({"points", "centers", "scores", "knn_idx", "output"})
    .Outputs({"out"})
    .Attrs({"B: int",
            "N0: int",
            "N1: int",
            "M: int",
            "K: int",
            "O: int",
            "aggregate: int"})
    .SetKernelFn(PD_KERNEL(assign_score_withk_forward_wrapper));
  
PD_BUILD_OP(assign_score_withk_backward_wrapper)
    .Inputs({"grad_out", "points", "centers", "scores", "knn_idx", "grad_points", "grad_centers", "grad_scores"})
    .Outputs({"out"})
    .Attrs({"B: int",
            "N0: int",
            "N1: int",
            "M: int",
            "K: int",
            "O: int",
            "aggregate: int"})
    .SetKernelFn(PD_KERNEL(assign_score_withk_backward_wrapper));