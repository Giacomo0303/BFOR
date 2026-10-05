import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, detection
from torchvision.models.detection.fcos import FCOSRegressionHead
from torchvision.ops import generalized_box_iou_loss


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


class BFOR_FCOS_Head(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        num_anchors: int = 1,
        num_convs: int = 4,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.box_coder = detection._utils.BoxLinearCoder(normalize_by_size=True)
        self.regression_head = FCOSRegressionHead(
            in_channels=in_channels, num_anchors=num_anchors, num_convs=num_convs
        )

    def forward(self, x):
        bbox_regression, bbox_objectness = self.regression_head(x)
        return {
            "cls_logits": bbox_objectness,
            "bbox_regression": bbox_regression,
            "bbox_ctrness": bbox_objectness,
        }

    def compute_loss(
        self,
        targets: list[dict[str, Tensor]],
        head_outputs: dict[str, Tensor],
        anchors: list[Tensor],
        matched_idxs: list[Tensor],
    ) -> dict[str, Tensor]:

        bbox_regression = head_outputs["bbox_regression"]  # [N, HWA, 4]
        bbox_ctrness = head_outputs["bbox_ctrness"]  # [N, HWA, 1]

        all_gt_classes_targets = []
        all_gt_boxes_targets = []
        for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
            if len(targets_per_image["labels"]) == 0:
                gt_classes_targets = targets_per_image["labels"].new_zeros(
                    (len(matched_idxs_per_image),)
                )
                gt_boxes_targets = targets_per_image["boxes"].new_zeros(
                    (len(matched_idxs_per_image), 4)
                )
            else:
                gt_classes_targets = targets_per_image["labels"][
                    matched_idxs_per_image.clip(min=0)
                ]
                gt_boxes_targets = targets_per_image["boxes"][
                    matched_idxs_per_image.clip(min=0)
                ]
            gt_classes_targets[matched_idxs_per_image < 0] = -1  # background
            all_gt_classes_targets.append(gt_classes_targets)
            all_gt_boxes_targets.append(gt_boxes_targets)

        # List[Tensor] to Tensor conversion of  `all_gt_boxes_target`, `all_gt_classes_targets` and `anchors`
        all_gt_boxes_targets, all_gt_classes_targets, anchors = (
            torch.stack(all_gt_boxes_targets),
            torch.stack(all_gt_classes_targets),
            torch.stack(anchors),
        )

        # compute foregroud
        foreground_mask = all_gt_classes_targets >= 0
        num_foreground = foreground_mask.sum().item()

        if num_foreground == 0:
            return {
                "bbox_regression": bbox_regression.sum() * 0.0,
                "bbox_objness": bbox_ctrness.sum() * 0.0,
            }

        # amp issue: pred_boxes need to convert float
        pred_boxes = self.box_coder.decode(bbox_regression, anchors)

        # regression loss: GIoU loss
        loss_bbox_reg = generalized_box_iou_loss(
            pred_boxes[foreground_mask],
            all_gt_boxes_targets[foreground_mask],
            reduction="sum",
        )

        # objness loss
        # Coordinate dei punti positivi
        anchor_centers = (anchors[:, :, :2] + anchors[:, :, 2:]) / 2.0
        pos_anchor_x = anchor_centers[foreground_mask][:, 0]
        pos_anchor_y = anchor_centers[foreground_mask][:, 1]

        # Box target corrispondenti [x0, y0, x1, y1]
        pos_gt = all_gt_boxes_targets[foreground_mask]
        cx = (pos_gt[:, 0] + pos_gt[:, 2]) / 2.0
        cy = (pos_gt[:, 1] + pos_gt[:, 3]) / 2.0
        w = (pos_gt[:, 2] - pos_gt[:, 0]).clamp(min=1.0)
        h = (pos_gt[:, 3] - pos_gt[:, 1]).clamp(min=1.0)

        # Soft target gaussiano B-FOR
        dx = (pos_anchor_x - cx) / w
        dy = (pos_anchor_y - cy) / h
        gt_objectness_targets = torch.exp(-1.0 / (2.0 * self.alpha) * (dx**2 + dy**2))

        # Loss BCE tra i logit predetti e il target gaussiano
        pred_objectness = head_outputs["bbox_ctrness"].squeeze(dim=2)
        loss_objectness = nn.functional.binary_cross_entropy_with_logits(
            pred_objectness[foreground_mask],
            gt_objectness_targets,
            reduction="sum",
        )

        return {
            "bbox_regression": loss_bbox_reg / max(1, num_foreground),
            "bbox_objness": loss_objectness / max(1, num_foreground),
        }


def build_bfor_fcos_model(alpha=0.1, pretrained_backbone=False, **kwargs):
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None
    head = BFOR_FCOS_Head(in_channels=256, num_anchors=1, num_convs=4, alpha=alpha)
    model = detection.fcos_resnet50_fpn(
        weights_backbone=weights_backbone,
        num_classes=1,
        head=head,
        **kwargs,
    )
    return model
