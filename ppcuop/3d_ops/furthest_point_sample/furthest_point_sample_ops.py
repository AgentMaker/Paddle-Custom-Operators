import os
__dirname__ = os.path.dirname(__file__)

from paddle.utils.cpp_extension import load

src_files = [
    'src/furthest_point_sample.cc', 'src/furthest_point_sample_cuda.cu'
]
src_files = [os.path.join(__dirname__, filename) for filename in src_files]

furthest_point_sample_ops = load(
    name="furthest_point_sample_ops",
    sources=src_files)

if __name__ == '__main__':
    furthest_point_sampling_wrapper_ops = furthest_point_sample_ops.furthest_point_sampling_wrapper
    print('furthest_point_sampling_wrapper_ops is passed!')
    furthest_point_sampling_with_dist_wrapper_ops = furthest_point_sample_ops.furthest_point_sampling_with_dist_wrapper
    print('furthest_point_sampling_with_dist_wrapper_ops is passed!')