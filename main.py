import torch

from src.model import FeaturePyramidNetwork, ResNet50Backbone, Encoder

model = ResNet50Backbone()
input = torch.randn(size=(1, 3, 448, 448))

fpn = FeaturePyramidNetwork()

enc = Encoder(model, fpn)

for out in enc(input):
    print(out.shape)