import torch
from torch import nn
import spconv as spconv_real

spconv_real.constants.SPCONV_USE_DIRECT_TABLE = False
import spconv.pytorch as spconv


class ResidualBlock(spconv.SparseModule):
    def __init__(
        self,
        inplanes,
        planes,
        downsample=None,
        bn_momentum=0.1,
        indice_key=None,
    ):
        super().__init__()
        self.first_block = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels=inplanes,
                out_channels=planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
            nn.BatchNorm1d(planes, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
            nn.BatchNorm1d(planes, momentum=bn_momentum),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.first_block(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out.replace_feature(out.features + residual.features)
        out = out.replace_feature(self.relu(out.features))

        return out


class Res16UNet34C(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        planes,
        layers,
        init_dim,
        conv1_kernel_size,
        bn_momentum,
    ):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.init_dim = init_dim
        self.conv1_kernel_size = conv1_kernel_size
        self.bn_momentum = bn_momentum

        self.network_initialization(in_channels)
        self.weight_initialization()

    def network_initialization(self, in_channels):
        self.inplanes = self.init_dim
        self.conv0p1s1 = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                self.inplanes,
                kernel_size=self.conv1_kernel_size,
                stride=1,
                padding=1,
                bias=False,
                indice_key="subm0",
            ),
            nn.BatchNorm1d(self.inplanes, momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
        )

        self.conv1p1s2 = spconv.SparseSequential(
            spconv.SparseConv3d(
                self.inplanes,
                self.inplanes,
                kernel_size=2,
                stride=2,
                padding=2,
                bias=False,
                indice_key="spconv1",
            ),
            nn.BatchNorm1d(self.inplanes, momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
            self._make_layer(
                self.planes[0],
                self.layers[0],
                bn_momentum=self.bn_momentum,
                indice_key="subm1",
            ),
        )

        self.conv2p2s2 = spconv.SparseSequential(
            spconv.SparseConv3d(
                self.inplanes,
                self.inplanes,
                kernel_size=2,
                stride=2,
                padding=2,
                indice_key="spconv2",
                bias=False,
            ),
            nn.BatchNorm1d(self.inplanes, momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
            self._make_layer(
                self.planes[1],
                self.layers[1],
                bn_momentum=self.bn_momentum,
                indice_key="subm2",
            ),
        )

        self.conv3p4s2 = spconv.SparseSequential(
            spconv.SparseConv3d(
                self.inplanes,
                self.inplanes,
                kernel_size=2,
                stride=2,
                padding=2,
                bias=False,
                indice_key="spconv3",
            ),
            nn.BatchNorm1d(self.inplanes, momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
            self._make_layer(
                self.planes[2],
                self.layers[2],
                bn_momentum=self.bn_momentum,
                indice_key="subm3",
            ),
        )

        self.conv4p8s2 = spconv.SparseSequential(
            spconv.SparseConv3d(
                self.inplanes,
                self.inplanes,
                kernel_size=2,
                stride=2,
                padding=2,
                bias=False,
                indice_key="spconv4",
            ),
            nn.BatchNorm1d(self.inplanes, momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
            self._make_layer(
                self.planes[3],
                self.layers[3],
                bn_momentum=self.bn_momentum,
                indice_key="subm4",
            ),
        )

        self.convtr4p16s2 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                self.inplanes,
                self.planes[4],
                kernel_size=2,
                bias=False,
                indice_key="spconv4",
            ),
            nn.BatchNorm1d(self.planes[4], momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.inplanes = self.planes[4] + self.planes[2]
        self.block5 = self._make_layer(
            self.planes[4],
            self.layers[4],
            bn_momentum=self.bn_momentum,
            indice_key="subm3",
        )

        self.convtr5p8s2 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                self.inplanes,
                self.planes[5],
                kernel_size=2,
                bias=False,
                indice_key="spconv3",
            ),
            nn.BatchNorm1d(self.planes[5], momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.inplanes = self.planes[5] + self.planes[1]
        self.block6 = self._make_layer(
            self.planes[5],
            self.layers[5],
            bn_momentum=self.bn_momentum,
            indice_key="subm2",
        )
        self.convtr6p4s2 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                self.inplanes,
                self.planes[6],
                kernel_size=2,
                bias=False,
                indice_key="spconv2",
            ),
            nn.BatchNorm1d(self.planes[6], momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.inplanes = self.planes[6] + self.planes[0]
        self.block7 = self._make_layer(
            self.planes[6],
            self.layers[6],
            bn_momentum=self.bn_momentum,
            indice_key="subm1",
        )
        self.convtr7p2s2 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                self.inplanes,
                self.planes[7],
                kernel_size=2,
                bias=False,
                indice_key="spconv1",
            ),
            nn.BatchNorm1d(self.planes[7], momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.inplanes = self.planes[7] + self.init_dim
        self.block8 = self._make_layer(
            self.planes[7],
            self.layers[7],
            bn_momentum=self.bn_momentum,
            indice_key="subm0",
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feature_maps = []
        out_p1 = self.conv0p1s1(x)

        out_b1p2 = self.conv1p1s2(out_p1)

        out_b2p4 = self.conv2p2s2(out_b1p2)

        out_b3p8 = self.conv3p4s2(out_b2p4)

        # pixel_dist=16
        out = self.conv4p8s2(out_b3p8)

        feature_maps.append(out)

        # pixel_dist=8
        out = self.convtr4p16s2(out)
        out = out.replace_feature(torch.hstack((out.features, out_b3p8.features)))
        out = self.block5(out)

        feature_maps.append(out)

        # pixel_dist=4
        out = self.convtr5p8s2(out)
        out = out.replace_feature(torch.hstack((out.features, out_b2p4.features)))
        out = self.block6(out)

        feature_maps.append(out)

        # pixel_dist=2
        out = self.convtr6p4s2(out)
        out = out.replace_feature(torch.hstack((out.features, out_b1p2.features)))
        out = self.block7(out)

        feature_maps.append(out)

        # pixel_dist=1
        out = self.convtr7p2s2(out)
        out = out.replace_feature(torch.hstack((out.features, out_p1.features)))
        out = self.block8(out)

        feature_maps.append(out)

        return feature_maps

    def _make_layer(self, planes, blocks, bn_momentum, indice_key):
        downsample = None
        if self.inplanes != planes:
            downsample = spconv.SparseSequential(
                spconv.SubMConv3d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    indice_key=indice_key,
                ),
                nn.BatchNorm1d(planes, momentum=bn_momentum),
            )
        layers = []
        layers.append(
            ResidualBlock(
                self.inplanes, planes, downsample=downsample, indice_key=indice_key
            )
        )
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes, indice_key=indice_key))
        return spconv.SparseSequential(*layers)
