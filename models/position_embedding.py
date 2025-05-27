import torch
from torch import nn
import numpy as np


def shift_scale_points(pred_xyz, input_range):
    """
    pred_xyz: B x N x 3
    input_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    dst_range = [
        torch.zeros_like(input_range[0], device=input_range[0].device),
        torch.ones_like(input_range[0], device=input_range[0].device),
    ]

    src_diff = input_range[1][:, None, :] - input_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - input_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz


class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        d_in=3,
        d_pos=None,
        normalize=True,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_pos = d_pos
        self.normalize = normalize

        # define a gaussian matrix input_ch -> output_ch
        B = torch.empty((d_in, d_pos // 2)).normal_()
        self.register_buffer("gauss_B", B)

    @torch.no_grad()
    def forward(self, xyz, num_channels=None, input_range=None):
        # xyz is batch x npoints x 3
        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        d_out = num_channels // 2

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        if self.normalize:
            xyz = shift_scale_points(xyz, input_range=input_range)

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, self.d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds
