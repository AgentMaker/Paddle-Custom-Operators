import paddle
from paddle.autograd import PyLayer

from .gather_points_ops import gather_points_ops
# from . import gather_points_ext


class GatherPoints(PyLayer):
    """Gather Points.

    Gather points with given index.
    """

    @staticmethod
    def forward(ctx, features: paddle.Tensor,
                indices: paddle.Tensor) -> paddle.Tensor:
        """forward.

        Args:
            features (Tensor): (B, C, N) features to gather.
            indices (Tensor): (B, M) where M is the number of points.

        Returns:
            Tensor: (B, C, M) where M is the number of points.
        """

        B, npoint = indices.shape
        _, C, N = features.shape
        output = paddle.zeros([B, C, npoint], dtype=paddle.float32)

        gather_points_ops.gather_points_wrapper(B, C, N, npoint, features,
                                                indices, output)

        ctx.save_for_backward(indices, C, N)
        indices.stop_gradient = True

        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.saved_tensor()
        B, npoint = idx.shape

        grad_features = paddle.zeros([B, C, N])
        grad_out_data = grad_out
        gather_points_ops.gather_points_grad_wrapper(B, C, N, npoint,
                                                     grad_out_data, idx,
                                                     grad_features.data)

        return grad_features, None


gather_points = GatherPoints.apply
