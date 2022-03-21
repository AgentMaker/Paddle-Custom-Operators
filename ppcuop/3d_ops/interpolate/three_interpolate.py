import paddle
from paddle.autograd import PyLayer
from typing import Tuple

from .interpolate_ops import interpolate_ops


class ThreeInterpolate(PyLayer):

    @staticmethod
    def forward(ctx, features: paddle.Tensor, indices: paddle.Tensor,
                weight: paddle.Tensor) -> paddle.Tensor:
        """Performs weighted linear interpolation on 3 features.

        Args:
            features (Tensor): (B, C, M) Features descriptors to be
                interpolated from
            indices (Tensor): (B, n, 3) index three nearest neighbors
                of the target features in features
            weight (Tensor): (B, n, 3) weights of interpolation

        Returns:
            Tensor: (B, C, N) tensor of the interpolated features
        """

        B, c, m = features.size()
        n = indices.size(1)
        ctx.three_interpolate_for_backward = (indices, weight, m)
        output = paddle.zeros((B, c, n), dtype=paddle.float32)

        interpolate_ops.three_interpolate_wrapper(B, c, m, n, features,
                                                  indices, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: paddle.Tensor):
        """Backward of three interpolate.

        Args:
            grad_out (Tensor): (B, C, N) tensor with gradients of outputs

        Returns:
            Tensor: (B, C, M) tensor with gradients of features
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = paddle.zeros((B, c, m), dtype=paddle.float32)
        grad_out_data = grad_out.data

        interpolate_ops.three_interpolate_grad_wrapper(B, c, n, m,
                                                       grad_out_data, idx,
                                                       weight,
                                                       grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply
