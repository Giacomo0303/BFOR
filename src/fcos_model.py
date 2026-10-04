import torchvision.models.detection as detection
from torchvision.models import ResNet50_Weights


def build_fcos_model(num_classes=21, pretrained_backbone=True, **kwargs):
    """
    Builds the standard FCOS model with ResNet-50-FPN backbone.

    Args:
        num_classes (int): Number of classes including background
                           (e.g. 21 for VOC: 20 foreground + 1 background).
        pretrained_backbone (bool): If True, uses official ImageNet-1K pretrained weights for ResNet-50.

    Returns:
        torchvision.models.detection.FCOS
    """
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None
    model = detection.fcos_resnet50_fpn(
        weights_backbone=weights_backbone,
        num_classes=num_classes,
        **kwargs,
    )
    return model
