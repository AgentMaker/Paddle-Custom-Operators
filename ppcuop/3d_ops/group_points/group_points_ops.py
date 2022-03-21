import os
__dirname__ = os.path.dirname(__file__)

from paddle.utils.cpp_extension import load

src_files = [
    'src/group_points.cc', 'src/group_points_cuda.cu'
]
src_files = [os.path.join(__dirname__, filename) for filename in src_files]

group_points_ops = load(
    name="group_points_ops",
    sources=src_files)

if __name__ == '__main__':
    group_points_grad_wrapper_ops = group_points_ops.group_points_grad_wrapper
    print('group_points_grad_wrapper_ops is passed!')
    group_points_wrapper_ops = group_points_ops.group_points_wrapper
    print('group_points_wrapper_ops is passed!')