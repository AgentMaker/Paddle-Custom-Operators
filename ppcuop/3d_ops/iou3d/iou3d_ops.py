import os
__dirname__ = os.path.dirname(__file__)

from paddle.utils.cpp_extension import load

src_files = [
    'src/iou3d.cc', 'src/iou3d_kernel.cu'
]
src_files = [os.path.join(__dirname__, filename) for filename in src_files]

iou3d_ops = load(
    name="iou3d_ops",
    sources=src_files)

if __name__ == '__main__':
    boxes_overlap_bev_gpu_ops = iou3d_ops.boxes_overlap_bev_gpu
    print('boxes_overlap_bev_gpu_ops is passed!')
    boxes_iou_bev_gpu_ops = iou3d_ops.boxes_iou_bev_gpu
    print('bev_gpu_ops is passed!')
    nms_gpu_ops = iou3d_ops.nms_gpu
    print('nms_gpu_ops is passed!')
    nms_normal_gpu_ops = iou3d_ops.nms_normal_gpu
    print('nms_normal_gpu_ops is passed!')