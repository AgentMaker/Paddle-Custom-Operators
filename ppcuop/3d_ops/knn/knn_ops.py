import os
__dirname__ = os.path.dirname(__file__)

from paddle.utils.cpp_extension import load

src_files = [
    'src/knn.cc', 'src/knn_cuda.cu'
]
src_files = [os.path.join(__dirname__, filename) for filename in src_files]

knn_ops = load(
    name="knn_ops",
    sources=src_files)

if __name__ == '__main__':
    knn_wrapper_ops = knn_ops.knn_wrapper
    print('knn_wrapper_ops is passed!')