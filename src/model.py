import torch.nn.functional as F
from torch import nn, sigmoid
from torchvision.models import resnet50


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(weights=None)

        self.layer0 = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
        )

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        sml_fm = self.layer2(x)
        med_fm = self.layer3(sml_fm)
        lrg_fm = self.layer4(med_fm)

        return sml_fm, med_fm, lrg_fm


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.lrg_conv1x1 = nn.Conv2d(
            in_channels=2048,
            out_channels=n_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        self.lrg_upsample = nn.Upsample(scale_factor=2)

        self.med_conv1x1 = nn.Conv2d(
            in_channels=1024,
            out_channels=n_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        self.med_upsample = nn.Upsample(scale_factor=2)

        self.sml_conv1x1 = nn.Conv2d(
            in_channels=512,
            out_channels=n_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        self.lrg_conv3x3 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.med_conv3x3 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.sml_conv3x3 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

    def forward(self, sml, med, lrg):
        lrg = self.lrg_conv1x1(lrg)
        med = self.lrg_upsample(lrg) + self.med_conv1x1(med)
        sml = self.med_upsample(med) + self.sml_conv1x1(sml)

        lrg_fm = self.lrg_conv3x3(lrg)
        med_fm = self.med_conv3x3(med)
        sml_fm = self.sml_conv3x3(sml)

        return sml_fm, med_fm, lrg_fm


class Encoder(nn.Module):
    def __init__(self, backbone, fpn, n_channels):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn

        self.lrg_conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.lrg_upsample = nn.Upsample(scale_factor=4)

        self.med_conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.med_upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        sml, med, lrg = self.backbone(x)
        sml_fm, med, lrg = self.fpn(sml, med, lrg)
        med_fm = self.med_upsample(self.med_conv(med))
        lrg_fm = self.lrg_upsample(self.lrg_conv(lrg))

        return sml_fm, med_fm, lrg_fm


class ObjHead(nn.Module):
    def __init__(self, n_channels, drop_rate):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.conv3x3_1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.conv3x3_2 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.conv1x1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = F.relu(self.conv_transpose(x))
        x = F.relu(self.conv3x3_1(x))
        x = F.relu(self.conv3x3_2(x))
        x = self.dropout(x)
        x = self.conv1x1(x)

        return sigmoid(x)


class ScaleHead(nn.Module):
    def __init__(self, n_channels, drop_rate):
        super().__init__()

        self.conv3x3_1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.conv3x3_2 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.dropout = nn.Dropout(p=drop_rate)

        self.conv3x3_3 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.conv1x1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding="same",
        )

    def forward(self, x):
        x = F.relu(self.conv3x3_1(x))
        x = F.relu(self.conv3x3_2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3x3_3(x))
        x = self.conv1x1(x)

        return sigmoid(x)


class Decoder(nn.Module):
    def __init__(self, n_channels, drop_rate):
        super().__init__()

        self.conv3x3_1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.conv_transpose_1 = nn.ConvTranspose2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.norm = nn.GroupNorm(num_groups=32, num_channels=n_channels)

        self.conv3x3_2 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.dropout = nn.Dropout(p=drop_rate)

        self.conv3x3_3 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.conv_transpose_2 = nn.ConvTranspose2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.conv_transpose_skip = nn.ConvTranspose2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.conv3x3_4 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.conv_transpose_3 = nn.ConvTranspose2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.conv3x3_5 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.obj_head = ObjHead(n_channels, drop_rate)
        self.width_head = ScaleHead(n_channels, drop_rate)
        self.height_head = ScaleHead(n_channels, drop_rate)

    def forward(self, x):
        x = F.relu(self.conv3x3_1(x))
        x = F.relu(self.conv_transpose_1(x))
        x_skip = F.relu(self.conv_transpose_skip(x))
        x = self.norm(x)
        x = F.relu(self.conv3x3_2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3x3_3(x))
        x = F.relu(self.conv_transpose_2(x)) + x_skip
        x = self.dropout(x)
        x = F.relu(self.conv3x3_4(x))

        obj_out = self.obj_head(x)

        x = F.relu(self.conv_transpose_3(x))
        x = F.relu(self.conv3x3_5(x))

        width_out = self.width_head(x)
        height_out = self.height_head(x)

        return obj_out, width_out, height_out


class BFOR_model(nn.Module):
    def __init__(self, n_channels, drop_rate):
        super().__init__()

        self.backbone = ResNet50Backbone()
        self.fpn = FeaturePyramidNetwork(n_channels=n_channels)
        self.encoder = Encoder(self.backbone, self.fpn, n_channels)

        self.sml_decoder = Decoder(n_channels, drop_rate)
        self.med_decoder = Decoder(n_channels, drop_rate)
        self.lrg_decoder = Decoder(n_channels, drop_rate)

    def forward(self, x):
        sml_fm, med_fm, lrg_fm = self.encoder(x)

        obj_sml, w_sml, h_sml = self.sml_decoder(sml_fm)
        obj_med, w_med, h_med = self.med_decoder(med_fm)
        obj_lrg, w_lrg, h_lrg = self.lrg_decoder(lrg_fm)

        return {
            "sml": {"obj": obj_sml, "w": w_sml, "h": h_sml},
            "med": {"obj": obj_med, "w": w_med, "h": h_med},
            "lrg": {"obj": obj_lrg, "w": w_lrg, "h": h_lrg},
        }
