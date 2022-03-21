import os
__dirname__ = os.path.dirname(__file__)

from paddle.utils.cpp_extension import load

src_files = [
    'src/ball_query.cc', 'src/ball_query_cuda.cu'
]
src_files = [os.path.join(__dirname__, filename) for filename in src_files]

ball_query_ops = load(
    name="ball_query_ops",
    sources=src_files)

if __name__ == '__main__':
    ball_query_wrapper_ops = ball_query_ops.ball_query_wrapper
    print('ball_query_wrapper_ops is passed!')