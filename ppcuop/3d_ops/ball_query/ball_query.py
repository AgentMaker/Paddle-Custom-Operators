import paddle
from paddle.autograd import PyLayer

from .ball_query_ops import ball_query_ops
#from . import ball_query_ext

class BallQuery(PyLayer):
    """Ball Query.

    Find nearby points in spherical space.
    """

    @staticmethod
    def forward(ctx, min_radius: float, max_radius: float, sample_num: int,
                xyz: paddle.Tensor, center_xyz: paddle.Tensor) -> paddle.Tensor:
        """forward.

        Args:
            min_radius (float): minimum radius of the balls.
            max_radius (float): maximum radius of the balls.
            sample_num (int): maximum number of features in the balls.
            xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            center_xyz (Tensor): (B, npoint, 3) centers of the ball query.

        Returns:
            Tensor: (B, npoint, nsample) tensor with the indicies of
                the features that form the query balls.
        """
        assert min_radius < max_radius

        B, N, _ = xyz.shape
        npoint = center_xyz.shape[1]

        idx = ball_query_ops.ball_query_wrapper(B, N, npoint, min_radius, max_radius,
                                                    sample_num, center_xyz, xyz)
        idx.stop_gradient = True
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply