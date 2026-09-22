import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 feature extractor outputting multi-scale features:
    C3: stride 8 (56x56, 512 channels)
    C4: stride 16 (28x28, 1024 channels)
    C5: stride 32 (14x14, 2048 channels)
    """

    def __init__(self, weights=None):
        super().__init__()
        base_model = resnet50(weights=weights)

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2  # C3: 56x56, 512 channels
        self.layer3 = base_model.layer3  # C4: 28x28, 1024 channels
        self.layer4 = base_model.layer4  # C5: 14x14, 2048 channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3, c4, c5


class FPN(nn.Module):
    """
    Feature Pyramid Network matching B-FOR authors' FPN:
    - Lateral 1x1 convolutions for C3, C4, C5
    - Bilinear 2x upsampling and top-down additive fusion
    - P3_conv, P4_conv, P5_conv (3x3 convs)
    - Scale homogenization bringing P3, P4, P5 all to 56x56:
        * p3 = P3 (56x56)
        * p4 = Conv2D -> ConvTranspose2D (stride 2) -> Conv2D (28x28 -> 56x56)
        * p5 = Conv2D -> ConvTranspose2D (stride 2) -> Conv2D -> ConvTranspose2D (stride 2) -> Conv2D (14x14 -> 56x56)
    """

    def __init__(self, out_channels=128):
        super().__init__()
        self.out_channels = out_channels

        self.lat_C3 = nn.Conv2d(512, out_channels, 1, padding="same")
        self.lat_C4 = nn.Conv2d(1024, out_channels, 1, padding="same")
        self.lat_C5 = nn.Conv2d(2048, out_channels, 1, padding="same")

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.P3_conv = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self.P4_conv = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self.P5_conv = nn.Conv2d(out_channels, out_channels, 3, padding="same")

        self.p4_conv1 = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self.upsample1 = nn.ConvTranspose2d(
            out_channels, out_channels, 3, stride=2, padding=1, output_padding=1
        )
        self.p4_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding="same")

        self.p5_conv1 = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self.upsample2 = nn.ConvTranspose2d(
            out_channels, out_channels, 3, stride=2, padding=1, output_padding=1
        )
        self.p5_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self.upsample3 = nn.ConvTranspose2d(
            out_channels, out_channels, 3, stride=2, padding=1, output_padding=1
        )
        self.p5_conv3 = nn.Conv2d(out_channels, out_channels, 3, padding="same")

    def forward(self, inputs):
        C3, C4, C5 = inputs

        L5 = self.lat_C5(C5)
        L4 = self.lat_C4(C4)
        L3 = self.lat_C3(C3)

        P5_td = L5
        P4_td = L4 + self.up2(P5_td)
        P3_td = L3 + self.up3(P4_td)

        P5 = self.P5_conv(P5_td)
        P4 = self.P4_conv(P4_td)
        P3 = self.P3_conv(P3_td)

        p3 = P3

        p4 = F.relu(self.p4_conv1(P4))
        p4 = F.relu(self.upsample1(p4))
        p4 = F.relu(self.p4_conv2(p4))

        p5 = F.relu(self.p5_conv1(P5))
        p5 = F.relu(self.upsample2(p5))
        p5 = F.relu(self.p5_conv2(p5))
        p5 = F.relu(self.upsample3(p5))
        p5 = F.relu(self.p5_conv3(p5))

        return p3, p4, p5


class HeadBlock(nn.Module):
    """
    Head / Decoder block exactly matching B-FOR authors' HeadBlock:
    - Input: 56x56 feature map (128 channels)
    - Shared Trunk with GroupNorm(32, 128) and residual skip connection:
        * conv1 (128->128, 3x3) -> up1 (ConvTranspose2d 3x3, stride 2, 56->112)
        * skip: skip_up (ConvTranspose2d 128->92, stride 2, 112->224)
        * norm1 (GroupNorm 32, 128) -> conv3 (128->92, 3x3) -> conv4 (92->92, 3x3) -> up3 (ConvTranspose2d 3x3, stride 2, 112->224)
        * add: x + skip (both 92 channels, 224x224)
        * conv5 (92->92, 3x3)
        * Scale branch: up4 (ConvTranspose2d 3x3, stride 2, 224->448) -> conv6 (92->92, 3x3)
        * Gaussian branch: up4_g (ConvTranspose2d 3x3, stride 2, 224->448) -> conv6_g (92->92, 3x3)
    - Gaussian (Objectness) Head:
        * g1 (92->92, 3x3) -> g2 (92->64, 3x3) -> g_out (64->1, 1x1, Sigmoid)
    - Height Head:
        * h8 -> h9 -> h10 -> h11 (all 92->92, 3x3) -> h_out (92->1, 1x1, ReLU)
    - Width Head:
        * w12 -> w13 -> w14 -> w15 (all 92->92, 3x3) -> w_out (92->1, 1x1, ReLU)
    """

    def __init__(self, drop_rate=0.0):
        super().__init__()
        # Shared trunk
        self.conv1 = nn.Conv2d(128, 128, 3, padding="same")
        self.up1 = nn.ConvTranspose2d(
            128, 128, 3, stride=2, padding=1, output_padding=1
        )

        self.skip_up = nn.ConvTranspose2d(
            128, 92, 3, stride=2, padding=1, output_padding=1
        )

        self.norm1 = nn.GroupNorm(32, 128)
        self.conv3 = nn.Conv2d(128, 92, 3, padding="same")
        self.drop1 = nn.Dropout(p=drop_rate)

        self.conv4 = nn.Conv2d(92, 92, 3, padding="same")
        self.up3 = nn.ConvTranspose2d(
            92, 92, 3, stride=2, padding=1, output_padding=1
        )

        self.drop2 = nn.Dropout(p=drop_rate)

        self.conv5 = nn.Conv2d(92, 92, 3, padding="same")
        self.up4 = nn.ConvTranspose2d(
            92, 92, 3, stride=2, padding=1, output_padding=1
        )
        self.up4_g = nn.ConvTranspose2d(
            92, 92, 3, stride=2, padding=1, output_padding=1
        )
        self.conv6 = nn.Conv2d(92, 92, 3, padding="same")
        self.conv6_g = nn.Conv2d(92, 92, 3, padding="same")

        # Gaussian head (objectness)
        self.g1 = nn.Conv2d(92, 92, 3, padding="same")
        self.gd = nn.Dropout(p=drop_rate)
        self.g2 = nn.Conv2d(92, 64, 3, padding="same")
        self.g_out = nn.Conv2d(64, 1, 1, padding="same")

        # Cov height
        self.h8 = nn.Conv2d(92, 92, 3, padding="same")
        self.h9 = nn.Conv2d(92, 92, 3, padding="same")
        self.hd = nn.Dropout(p=drop_rate)
        self.h10 = nn.Conv2d(92, 92, 3, padding="same")
        self.h11 = nn.Conv2d(92, 92, 3, padding="same")
        self.h_out = nn.Conv2d(92, 1, 1, padding="same")

        # Cov width
        self.w12 = nn.Conv2d(92, 92, 3, padding="same")
        self.w13 = nn.Conv2d(92, 92, 3, padding="same")
        self.wd = nn.Dropout(p=drop_rate)
        self.w14 = nn.Conv2d(92, 92, 3, padding="same")
        self.w15 = nn.Conv2d(92, 92, 3, padding="same")
        self.w_out = nn.Conv2d(92, 1, 1, padding="same")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.up1(x))

        skip = x
        x = self.norm1(x)
        x = F.relu(self.conv3(x))
        skip = F.relu(self.skip_up(skip))

        x = self.drop1(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.up3(x))

        x = x + skip
        x = self.drop2(x)

        x = F.relu(self.conv5(x))
        feat_cov = F.relu(self.up4(x))
        shared = F.relu(self.conv6(feat_cov))
        feat_g = F.relu(self.up4_g(x))
        feat_g = F.relu(self.conv6_g(feat_g))

        # Gaussian / Objectness
        g = F.relu(self.g1(feat_g))
        g = self.gd(g)
        g = F.relu(self.g2(g))
        g = torch.sigmoid(self.g_out(g))

        # Cov height
        h = F.relu(self.h8(shared))
        h = F.relu(self.h9(h))
        h = self.hd(h)
        h = F.relu(self.h10(h))
        h = F.relu(self.h11(h))
        ch = F.relu(self.h_out(h))

        # Cov width
        w = F.relu(self.w12(shared))
        w = F.relu(self.w13(w))
        w = self.wd(w)
        w = F.relu(self.w14(w))
        w = F.relu(self.w15(w))
        cw = F.relu(self.w_out(w))

        return g, ch, cw


class BFOR_model(nn.Module):
    """
    B-FOR detection model matching authors' CustomFPNModel:
    - FeatureExtractor: ResNet-50 backbone (C3, C4, C5)
    - FPN: Multi-scale Feature Pyramid Network with learned upsamplers to 56x56
    - head_P3, head_P4, head_P5: HeadBlock decoders for small, medium, and large scales
    """

    def __init__(self, n_channels=128, drop_rate=0.0, backbone_weights=None):
        super().__init__()
        self.feature_extractor = ResNet50Backbone(weights=backbone_weights)
        self.fpn = FPN(out_channels=n_channels)
        self.head_P3 = HeadBlock(drop_rate=drop_rate)
        self.head_P4 = HeadBlock(drop_rate=drop_rate)
        self.head_P5 = HeadBlock(drop_rate=drop_rate)

        # Aliases for backward compatibility
        self.backbone = self.feature_extractor
        self.sml_decoder = self.head_P3
        self.med_decoder = self.head_P4
        self.lrg_decoder = self.head_P5

    def forward(self, x):
        c3, c4, c5 = self.feature_extractor(x)
        p3, p4, p5 = self.fpn((c3, c4, c5))

        g3, ch3, cw3 = self.head_P3(p3)
        g4, ch4, cw4 = self.head_P4(p4)
        g5, ch5, cw5 = self.head_P5(p5)

        return {
            "sml": {
                "obj": g3,
                "w": cw3,
                "h": ch3,
                "gaussian": g3,
                "cov_h": ch3,
                "cov_w": cw3,
            },
            "med": {
                "obj": g4,
                "w": cw4,
                "h": ch4,
                "gaussian": g4,
                "cov_h": ch4,
                "cov_w": cw4,
            },
            "lrg": {
                "obj": g5,
                "w": cw5,
                "h": ch5,
                "gaussian": g5,
                "cov_h": ch5,
                "cov_w": cw5,
            },
            "P3": {
                "obj": g3,
                "w": cw3,
                "h": ch3,
                "gaussian": g3,
                "cov_h": ch3,
                "cov_w": cw3,
            },
            "P4": {
                "obj": g4,
                "w": cw4,
                "h": ch4,
                "gaussian": g4,
                "cov_h": ch4,
                "cov_w": cw4,
            },
            "P5": {
                "obj": g5,
                "w": cw5,
                "h": ch5,
                "gaussian": g5,
                "cov_h": ch5,
                "cov_w": cw5,
            },
        }


# Aliases
FeaturePyramidNetwork = FPN
Decoder = HeadBlock
CustomFPNModel = BFOR_model
