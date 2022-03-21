// Modified from
// https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp

/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <paddle/extension.h>

#include <cstdint>
#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_ERROR(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

void boxesoverlapLauncher(const int num_a, const float *boxes_a,
                          const int num_b, const float *boxes_b,
                          float *ans_overlap);
void boxesioubevLauncher(const int num_a, const float *boxes_a, const int num_b,
                         const float *boxes_b, float *ans_iou);
void nmsLauncher(const float *boxes, unsigned long long *mask, int boxes_num,
                 float nms_overlap_thresh);
void nmsNormalLauncher(const float *boxes, unsigned long long *mask,
                       int boxes_num, float nms_overlap_thresh);

std::vector<paddle::Tensor> constant_tensor(int constant){
  auto constant_tensor = paddle::Tensor(paddle::PlaceType::kCPU, std::vector<int64_t> {1});
  int* data = constant_tensor.mutable_data<int>(paddle::PlaceType::kCPU);
  data[0] = constant;
  return {constant_tensor};
}

std::vector<paddle::Tensor> boxes_overlap_bev_gpu(const paddle::Tensor& boxes_a, const paddle::Tensor& boxes_b,
                                                  const paddle::Tensor& ans_overlap) {
  // params boxes_a: (N, 5) [x1, y1, x2, y2, ry]
  // params boxes_b: (M, 5)
  // params ans_overlap: (N, M)

  CHECK_INPUT(boxes_a);
  CHECK_INPUT(boxes_b);
  CHECK_INPUT(ans_overlap);

  int num_a = boxes_a.shape()[0];
  int num_b = boxes_b.shape()[0];

  const float *boxes_a_data = boxes_a.data<float>();
  const float *boxes_b_data = boxes_b.data<float>();
  float *ans_overlap_data = ans_overlap.data<float>();

  boxesoverlapLauncher(num_a, boxes_a_data, num_b, boxes_b_data,
                       ans_overlap_data);

  return constant_tensor(1);
}

std::vector<paddle::Tensor> boxes_iou_bev_gpu(const paddle::Tensor& boxes_a, const paddle::Tensor& boxes_b,
                                              const paddle::Tensor& ans_iou) {
  // params boxes_a: (N, 5) [x1, y1, x2, y2, ry]
  // params boxes_b: (M, 5)
  // params ans_overlap: (N, M)

  CHECK_INPUT(boxes_a);
  CHECK_INPUT(boxes_b);
  CHECK_INPUT(ans_iou);

  int num_a = boxes_a.shape()[0];
  int num_b = boxes_b.shape()[0];

  const float *boxes_a_data = boxes_a.data<float>();
  const float *boxes_b_data = boxes_b.data<float>();
  float *ans_iou_data = const_cast<float*>(ans_iou.data<float>());

  boxesioubevLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_iou_data);

  return constant_tensor(1);
}

std::vector<paddle::Tensor> nms_gpu(const paddle::Tensor& boxes, const paddle::Tensor& keep,
                                    float nms_overlap_thresh, int device_id) {
  // params boxes: (N, 5) [x1, y1, x2, y2, ry]
  // params keep: (N)

  CHECK_INPUT(boxes);
  cudaSetDevice(device_id);

  int boxes_num = boxes.shape()[0];
  const float *boxes_data = boxes.data<float>();
  int64_t *keep_data = const_cast<int64_t*>(keep.data<int64_t>());

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  unsigned long long *mask_data = NULL;
  CHECK_ERROR(cudaMalloc((void **)&mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long)));
  nmsLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

  // unsigned long long mask_cpu[boxes_num * col_blocks];
  // unsigned long long *mask_cpu = new unsigned long long [boxes_num *
  // col_blocks];
  std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

  //    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
  CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));

  cudaFree(mask_data);

  unsigned long long *remv_cpu = new unsigned long long[col_blocks]();

  int num_to_keep = 0;

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      unsigned long long *p = &mask_cpu[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }
  delete[] remv_cpu;
  if (cudaSuccess != cudaGetLastError()) printf("Error!\n");

  return constant_tensor(num_to_keep);
}

std::vector<paddle::Tensor> nms_normal_gpu(const paddle::Tensor& boxes, const paddle::Tensor& keep,
                   float nms_overlap_thresh, int device_id) {
  // params boxes: (N, 5) [x1, y1, x2, y2, ry]
  // params keep: (N)

  CHECK_INPUT(boxes);
  cudaSetDevice(device_id);

  int boxes_num = boxes.shape()[0];
  const float *boxes_data = boxes.data<float>();
  int64_t *keep_data = const_cast<int64_t*>(keep.data<int64_t>());

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  unsigned long long *mask_data = NULL;
  CHECK_ERROR(cudaMalloc((void **)&mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long)));
  nmsNormalLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

  // unsigned long long mask_cpu[boxes_num * col_blocks];
  // unsigned long long *mask_cpu = new unsigned long long [boxes_num *
  // col_blocks];
  std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

  //    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
  CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));

  cudaFree(mask_data);

  unsigned long long *remv_cpu = new unsigned long long[col_blocks]();

  int num_to_keep = 0;

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      unsigned long long *p = &mask_cpu[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }
  delete[] remv_cpu;
  if (cudaSuccess != cudaGetLastError()) printf("Error!\n");

  return constant_tensor(num_to_keep);
}

PD_BUILD_OP(boxes_overlap_bev_gpu)
    .Inputs({"boxes_a", "boxes_b", "ans_overlap"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(boxes_overlap_bev_gpu));

PD_BUILD_OP(boxes_iou_bev_gpu)
    .Inputs({"boxes_a", "boxes_b", "ans_iou"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(boxes_iou_bev_gpu));

PD_BUILD_OP(nms_gpu)
    .Inputs({"boxes", "keep"})
    .Outputs({"out"})
    .Attrs({"nms_overlap_thresh: float",
            "device_id: int"})
    .SetKernelFn(PD_KERNEL(nms_gpu));

PD_BUILD_OP(nms_normal_gpu)
    .Inputs({"boxes", "keep"})
    .Outputs({"out"})
    .Attrs({"nms_overlap_thresh: float",
            "device_id: int"})
    .SetKernelFn(PD_KERNEL(nms_normal_gpu));
