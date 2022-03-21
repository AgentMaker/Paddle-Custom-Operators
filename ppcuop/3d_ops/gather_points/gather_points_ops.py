import os
__dirname__ = os.path.dirname(__file__)

from paddle.utils.cpp_extension import load

src_files = [
    'src/gather_points.cc', 'src/gather_points_cuda.cu'
]
src_files = [os.path.join(__dirname__, filename) for filename in src_files]

gather_points_ops = load(
    name="gather_points_ops",
    sources=src_files)

if __name__ == '__main__':
    gather_points_wrapper_ops = gather_points_ops.gather_points_wrapper
    print('gather_points_wrapper_ops is passed!')
    gather_points_grad_wrapper_ops = gather_points_ops.gather_points_grad_wrapper
    print('gather_points_grad_wrapper_ops is passed!')