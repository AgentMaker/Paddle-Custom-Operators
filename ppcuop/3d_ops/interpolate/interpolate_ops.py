import os
__dirname__ = os.path.dirname(__file__)

from paddle.utils.cpp_extension import load

src_files = [
    'src/interpolate.cc', 'src/three_interpolate_cuda.cu', 'src/three_nn_cuda.cu'
]
src_files = [os.path.join(__dirname__, filename) for filename in src_files]

interpolate_ops = load(
    name="interpolate_ops",
    sources=src_files)

if __name__ == '__main__':
    three_nn_wrapper_ops = interpolate_ops.three_nn_wrapper
    print('three_nn_wrapper_ops is passed!')
    three_interpolate_wrapper_ops = interpolate_ops.three_interpolate_wrapper
    print('three_interpolate_wrapper_ops is passed!')
    three_interpolate_grad_wrapper_ops = interpolate_ops.three_interpolate_grad_wrapper
    print('three_interpolate_grad_wrapper_ops is passed!')