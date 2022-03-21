import paddle
from paddle.autograd import PyLayer

from .furthest_point_sample_ops import furthest_point_sample_ops
# from . import furthest_point_sample_ext

class FurthestPointSampling(PyLayer):
    """Furthest Point Sampling.

    Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance.
    """

    @staticmethod
    def forward(ctx, points_xyz: paddle.Tensor,
                num_points: int) -> paddle.Tensor:
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) where N > num_points.
            num_points (int): Number of points in the sampled set.

        Returns:
             Tensor: (B, num_points) indices of the sampled points.
        """

        B, N = points_xyz.shape[:2]
        output = paddle.zeros([B, num_points], dtype=paddle.int32)
        temp = paddle.full([B, N], 1e10, dtype=paddle.float32)

        furthest_point_sample_ops.furthest_point_sampling_wrapper(
            B, N, num_points, points_xyz, temp, output)
        output.stop_gradient = True

        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


class FurthestPointSamplingWithDist(PyLayer):
    """Furthest Point Sampling With Distance.

    Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance.
    """

    @staticmethod
    def forward(ctx, points_dist: paddle.Tensor,
                num_points: int) -> paddle.Tensor:
        """forward.

        Args:
            points_dist (Tensor): (B, N, N) Distance between each point pair.
            num_points (int): Number of points in the sampled set.

        Returns:
             Tensor: (B, num_points) indices of the sampled points.
        """

        B, N, _ = points_dist.shape
        output = paddle.zeros([B, num_points], dtype=paddle.int32)
        temp = paddle.full([B, N], 1e10, dtype=paddle.float32)

        furthest_point_sample_ops.furthest_point_sampling_with_dist_wrapper(
            B, N, num_points, points_dist, temp, output)
        output.stop_gradient = True

        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply
furthest_point_sample_with_dist = FurthestPointSamplingWithDist.apply
