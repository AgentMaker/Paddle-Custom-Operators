// Modified from
// https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu
// Written by Shaoshuai Shi
// All Rights Reserved 2019.

#include <assert.h>
#include <paddle/extension.h>

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

void points_in_boxes_launcher(int batch_size, int boxes_num, int pts_num,
                              const float *boxes, const float *pts,
                              int *box_idx_of_points);

void points_in_boxes_batch_launcher(int batch_size, int boxes_num, int pts_num,
                                    const float *boxes, const float *pts,
                                    int *box_idx_of_points);

void roiaware_pool3d_launcher(int boxes_num, int pts_num, int channels,
                              int max_pts_each_voxel, int out_x, int out_y,
                              int out_z, const float *rois, const float *pts,
                              const float *pts_feature, int *argmax,
                              int *pts_idx_of_voxels, float *pooled_features,
                              int pool_method);

void roiaware_pool3d_backward_launcher(int boxes_num, int out_x, int out_y,
                                       int out_z, int channels,
                                       int max_pts_each_voxel,
                                       const int *pts_idx_of_voxels,
                                       const int *argmax, const float *grad_out,
                                       float *grad_in, int pool_method);

int points_in_boxes_cpu(const paddle::Tensor& boxes_tensor, const paddle::Tensor& pts_tensor,
                        const paddle::Tensor& pts_indices_tensor);

int points_in_boxes_gpu(const paddle::Tensor& boxes_tensor, const paddle::Tensor& pts_tensor,
                        const paddle::Tensor& box_idx_of_points_tensor) {
  // params boxes: (B, N, 7) [x, y, z, w, l, h, rz] in LiDAR coordinate, z is
  // the bottom center, each box DO NOT overlaps params pts: (B, npoints, 3) [x,
  // y, z] in LiDAR coordinate params boxes_idx_of_points: (B, npoints), default
  // -1

  CHECK_INPUT(boxes_tensor);
  CHECK_INPUT(pts_tensor);
  CHECK_INPUT(box_idx_of_points_tensor);

  int batch_size = boxes_tensor.shape[0];
  int boxes_num = boxes_tensor.shape[1];
  int pts_num = pts_tensor.shape[1];

  const float *boxes = boxes_tensor.data<float>();
  const float *pts = pts_tensor.data<float>();
  int *box_idx_of_points = const_cast<int*>(box_idx_of_points_tensor.data<int>());

  points_in_boxes_launcher(batch_size, boxes_num, pts_num, boxes, pts,
                           box_idx_of_points);

  return 1;
}

int points_in_boxes_batch(const paddle::Tensor& boxes_tensor, const paddle::Tensor& pts_tensor,
                          const paddle::Tensor& box_idx_of_points_tensor) {
  // params boxes: (B, N, 7) [x, y, z, w, l, h, rz] in LiDAR coordinate, z is
  // the bottom center. params pts: (B, npoints, 3) [x, y, z] in LiDAR
  // coordinate params boxes_idx_of_points: (B, npoints), default -1

  CHECK_INPUT(boxes_tensor);
  CHECK_INPUT(pts_tensor);
  CHECK_INPUT(box_idx_of_points_tensor);

  int batch_size = boxes_tensor.shape[0];
  int boxes_num = boxes_tensor.shape[1];
  int pts_num = pts_tensor.shape[1];

  const float *boxes = boxes_tensor.data<float>();
  const float *pts = pts_tensor.data<float>();
  int *box_idx_of_points = const_cast<int*>(box_idx_of_points_tensor.data<int>());

  points_in_boxes_batch_launcher(batch_size, boxes_num, pts_num, boxes, pts,
                                 box_idx_of_points);

  return 1;
}

int roiaware_pool3d_gpu(const paddle::Tensor& rois, const paddle::Tensor& pts, const paddle::Tensor& pts_feature,
                        const paddle::Tensor& argmax, const paddle::Tensor& pts_idx_of_voxels,
                        const paddle::Tensor& pooled_features, int pool_method) {
  // params rois: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coordinate
  // params pts: (npoints, 3) [x, y, z] in LiDAR coordinate
  // params pts_feature: (npoints, C)
  // params argmax: (N, out_x, out_y, out_z, C)
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  // params pooled_features: (N, out_x, out_y, out_z, C)
  // params pool_method: 0: max_pool 1: avg_pool

  CHECK_INPUT(rois);
  CHECK_INPUT(pts);
  CHECK_INPUT(pts_feature);
  CHECK_INPUT(argmax);
  CHECK_INPUT(pts_idx_of_voxels);
  CHECK_INPUT(pooled_features);

  int boxes_num = rois.shape[0];
  int pts_num = pts.shape[0];
  int channels = pts_feature.shape[1];
  int max_pts_each_voxel = pts_idx_of_voxels.shape[4];  // index 0 is the counter
  int out_x = pts_idx_of_voxels.shape[1];
  int out_y = pts_idx_of_voxels.shape[2];
  int out_z = pts_idx_of_voxels.shape[3];
  assert((out_x < 256) && (out_y < 256) &&
         (out_z < 256));  // we encode index with 8bit

  const float *rois_data = rois.data<float>();
  const float *pts_data = pts.data<float>();
  const float *pts_feature_data = pts_feature.data<float>();
  int *argmax_data = const_cast<int*>(argmax.data<int>());
  int *pts_idx_of_voxels_data = const_cast<int*>(pts_idx_of_voxels.data<int>());
  float *pooled_features_data = const_cast<float*>(pooled_features.data<float>());

  roiaware_pool3d_launcher(
      boxes_num, pts_num, channels, max_pts_each_voxel, out_x, out_y, out_z,
      rois_data, pts_data, pts_feature_data, argmax_data,
      pts_idx_of_voxels_data, pooled_features_data, pool_method);

  return 1;
}

int roiaware_pool3d_gpu_backward(const paddle::Tensor& pts_idx_of_voxels,
                                 const paddle::Tensor& argmax, const paddle::Tensor& grad_out,
                                 const paddle::Tensor& grad_in, int pool_method) {
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  // params argmax: (N, out_x, out_y, out_z, C)
  // params grad_out: (N, out_x, out_y, out_z, C)
  // params grad_in: (npoints, C), return value
  // params pool_method: 0: max_pool 1: avg_pool

  CHECK_INPUT(pts_idx_of_voxels);
  CHECK_INPUT(argmax);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(grad_in);

  int boxes_num = pts_idx_of_voxels.shape[0];
  int out_x = pts_idx_of_voxels.shape[1];
  int out_y = pts_idx_of_voxels.shape[2];
  int out_z = pts_idx_of_voxels.shape[3];
  int max_pts_each_voxel = pts_idx_of_voxels.shape[4];  // index 0 is the counter
  int channels = grad_out.shape[4];

  const int *pts_idx_of_voxels_data = pts_idx_of_voxels.data<int>();
  const int *argmax_data = argmax.data<int>();
  const float *grad_out_data = grad_out.data<float>();
  float *grad_in_data = const_cast<float*>(grad_in.data<float>());

  roiaware_pool3d_backward_launcher(boxes_num, out_x, out_y, out_z, channels,
                                    max_pts_each_voxel, pts_idx_of_voxels_data,
                                    argmax_data, grad_out_data, grad_in_data,
                                    pool_method);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &roiaware_pool3d_gpu, "roiaware pool3d forward (CUDA)");
  m.def("backward", &roiaware_pool3d_gpu_backward,
        "roiaware pool3d backward (CUDA)");
  m.def("points_in_boxes_gpu", &points_in_boxes_gpu,
        "points_in_boxes_gpu forward (CUDA)");
  m.def("points_in_boxes_batch", &points_in_boxes_batch,
        "points_in_boxes_batch forward (CUDA)");
  m.def("points_in_boxes_cpu", &points_in_boxes_cpu,
        "points_in_boxes_cpu forward (CPU)");
}

PD_BUILD_OP(roiaware_pool3d_gpu)
    .Inputs({"rois", "pts", "pts_feature", "argmax", "pts_idx_of_voxels", "pooled_features"})
    .Outputs({"out"})
    .Attrs({"pool_method: int"})
    .SetKernelFn(PD_KERNEL(roiaware_pool3d_gpu));

PD_BUILD_OP(roiaware_pool3d_gpu_backward)
    .Inputs({"pts_idx_of_voxels", "argmax", "grad_out", "grad_in"})
    .Outputs({"out"})
    .Attrs({"pool_method: int"})
    .SetKernelFn(PD_KERNEL(roiaware_pool3d_gpu_backward));
    
PD_BUILD_OP(points_in_boxes_gpu)
    .Inputs({"boxes_tensor", "pts_tensor", "box_idx_of_points_tensor"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(points_in_boxes_gpu));

PD_BUILD_OP(points_in_boxes_batch)
    .Inputs({"boxes_tensor", "pts_tensor", "box_idx_of_points_tensor"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(points_in_boxes_batch));

PD_BUILD_OP(points_in_boxes_cpu)
    .Inputs({"boxes_tensor", "pts_tensor", "box_idx_of_points_tensor"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(points_in_boxes_cpu));