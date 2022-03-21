import paddle
from paddle.autograd import PyLayer
from typing import Tuple

from .interpolate_ops import interpolate_ops


class ThreeNN(PyLayer):

    @staticmethod
    def forward(ctx, target: paddle.Tensor,
                source: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Find the top-3 nearest neighbors of the target set from the source
        set.

        Args:
            target (Tensor): shape (B, N, 3), points set that needs to
                find the nearest neighbors.
            source (Tensor): shape (B, M, 3), points set that is used
                to find the nearest neighbors of points in target set.

        Returns:
            Tensor: shape (B, N, 3), L2 distance of each point in target
                set to their corresponding nearest neighbors.
        """

        B, N, _ = target.size()
        m = source.size(1)
        dist2 = paddle.zeros((B, N, 3), dtype=paddle.float32)
        idx = paddle.zeros((B, N, 3), dtype=paddle.int64)

        interpolate_ops.three_nn_wrapper(B, N, m, target, source, dist2, idx)

        idx.stop_gradient = True

        return paddle.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply
