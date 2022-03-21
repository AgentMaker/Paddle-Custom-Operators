import paddle

def calc_square_dist(point_feat_a, point_feat_b, norm=True):
    """Calculating square distance between a and b.

    Args:
        point_feat_a (Tensor): (B, N, C) Feature vector of each point.
        point_feat_b (Tensor): (B, M, C) Feature vector of each point.
        norm (Bool): Whether to normalize the distance.
            Default: True.

    Returns:
        Tensor: (B, N, M) Distance between each pair points.
    """
    length_a = point_feat_a.shape[1]
    length_b = point_feat_b.shape[1]
    num_channel = point_feat_a.shape[-1]
    # [bs, n, 1]
    a_square = paddle.sum(point_feat_a.unsqueeze(2).pow(2), axis=-1)
    # [bs, 1, m]
    b_square = paddle.sum(point_feat_b.unsqueeze(1).pow(2), axis=-1)
    a_square = paddle.tile(a_square, (1, 1, length_b))  # [bs, n, m]
    b_square = paddle.tile(b_square, (1, length_a, 1))  # [bs, n, m]

    coor = paddle.matmul(point_feat_a, point_feat_b.transpose((1, 2)))

    dist = a_square + b_square - 2 * coor
    if norm:
        dist = paddle.sqrt(dist) / num_channel
    return dist
