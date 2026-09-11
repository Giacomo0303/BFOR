import torch
from torch import nn
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
    def __init__(self):
        super().__init__()

        self.lrg_conv1x1 = nn.Conv2d(
            in_channels=2048, out_channels=128, kernel_size=1, stride=1, padding="same"
        )

        self.lrg_upsample = nn.Upsample(scale_factor=2)

        self.med_conv1x1 = nn.Conv2d(
            in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding="same"
        )

        self.med_upsample = nn.Upsample(scale_factor=2)

        self.sml_conv1x1 = nn.Conv2d(
            in_channels=512, out_channels=128, kernel_size=1, stride=1, padding="same"
        )

        self.lrg_conv3x3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same"
        )

        self.med_conv3x3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same"
        )

        self.sml_conv3x3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same"
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
    def __init__(self, backbone, fpn):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn

        self.lrg_conv = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same"
        )
        self.lrg_upsample = nn.Upsample(scale_factor=4)

        self.med_conv = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same"
        )
        self.med_upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        sml, med, lrg = self.backbone(x)
        sml_fm, med, lrg = self.fpn(sml, med, lrg)
        med_fm = self.med_upsample(self.med_conv(med))
        lrg_fm = self.lrg_upsample(self.lrg_conv(lrg))

        return sml_fm, med_fm, lrg_fm

        

