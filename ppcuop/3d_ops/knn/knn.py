import paddle
from paddle.autograd import PyLayer

from .knn_ops import knn_ops


class KNN(PyLayer):
    r"""KNN (CUDA) based on heap data structure.
    Modified from `PAConv <https://github.com/CVMI-Lab/PAConv/tree/main/
    scene_seg/lib/pointops/src/knnquery_heap>`_.

    Find k-nearest points.
    """

    @staticmethod
    def forward(ctx,
                k: int,
                xyz: paddle.Tensor,
                center_xyz: paddle.Tensor = None,
                transposed: bool = False) -> paddle.Tensor:
        """Forward.

        Args:
            k (int): number of nearest neighbors.
            xyz (Tensor): (B, N, 3) if transposed == False, else (B, 3, N).
                xyz coordinates of the features.
            center_xyz (Tensor): (B, npoint, 3) if transposed == False,
                else (B, 3, npoint). centers of the knn query.
            transposed (bool): whether the input tensors are transposed.
                defaults to False. Should not expicitly use this keyword
                when calling knn (=KNN.apply), just add the fourth param.

        Returns:
            Tensor: (B, k, npoint) tensor with the indicies of
                the features that form k-nearest neighbours.
        """
        assert k > 0

        if center_xyz is None:
            center_xyz = xyz

        if transposed:
            xyz = xyz.transpose((2, 1))
            center_xyz = center_xyz.transpose((2, 1))

        center_xyz_device = center_xyz.get_device()
        assert center_xyz_device == xyz.get_device(), \
            'center_xyz and xyz should be put on the same device'

        B, npoint, _ = center_xyz.shape
        N = xyz.shape[1]

        idx = center_xyz.new_zeros((B, npoint, k)).int()
        dist2 = center_xyz.new_zeros((B, npoint, k)).float()

        knn_ops.knn_wrapper(B, N, npoint, k, xyz, center_xyz, idx, dist2)
        # idx shape to [B, k, npoint]
        idx = idx.transpose((2, 1))
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


knn = KNN.apply
