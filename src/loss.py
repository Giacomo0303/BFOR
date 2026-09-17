import torch
from torch import nn


class BFOR_Loss(nn.Module):
    def __init__(self, alpha, lambda_ctr, k, device):
        super().__init__()

        self.alpha = alpha
        self.lambda_ctr = lambda_ctr
        self.k = k
        self.device = device

    # preds should be a dictionary
    # {"sml" or "med" or "lrg": {obj: [B, 1, 448, 448], w: [B, 1, 448, 448], h: [B, 1, 448, 448]}
    # targets should be a list of tensors of shape [N_box, 4]
    def forward(self, preds, targets):
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            total_loss = 0.0
            valid_images = 0

            for b, target_boxes in enumerate(targets):
                if target_boxes.shape[0] > 0:
                    img_loss = self.compute_image_loss(
                        img_num=b, preds=preds, tgt_boxes=target_boxes
                    )
                    if valid_images == 0:
                        total_loss = img_loss
                    else:
                        total_loss = total_loss + img_loss
                    valid_images += 1

            if valid_images == 0:
                return torch.tensor(0.0, device=self.device)

            return total_loss / valid_images

    def compute_image_loss(self, img_num, preds, tgt_boxes):
        num_boxes = tgt_boxes.shape[0]

        if num_boxes == 0:
            return torch.tensor(0.0, device=self.device)

        img_loss = 0.0

        for i in range(num_boxes):
            tgt_box = tgt_boxes[i, :] * 448  # denormalize coords
            scale = self.get_scale(box=tgt_box)
            pred = {
                "obj": preds[scale]["obj"][img_num, ...].float(),
                "w": preds[scale]["w"][img_num, ...].float(),
                "h": preds[scale]["h"][img_num, ...].float(),
            }

            obj_loss, pred_ctr = self.obj_loss(pred, tgt_box)
            scale_loss = self.scale_loss(pred, tgt_box, pred_ctr)

            if i == 0:
                img_loss = obj_loss + scale_loss
            else:
                img_loss = img_loss + obj_loss + scale_loss

        img_loss = img_loss / num_boxes

        return img_loss

    def get_scale(self, box):
        # box is [c_x, c_y, w, h]
        size = torch.sqrt(box[2] * box[3])

        if size < 32.0:
            return "sml"
        elif size >= 96.0:
            return "lrg"
        else:
            return "med"

    def obj_loss(self, pred, tgt_box):
        # find the limits of the tgt box
        cx, cy, w, h = tgt_box
        w = max(float(w), 1.0)
        h = max(float(h), 1.0)

        x1 = max(0, min(447, int(cx - w / 2)))
        x2 = max(x1 + 1, min(448, int(cx + w / 2)))
        y1 = max(0, min(447, int(cy - h / 2)))
        y2 = max(y1 + 1, min(448, int(cy + h / 2)))

        # meshgrid
        x_range = torch.arange(x1, x2, device=self.device)
        y_range = torch.arange(y1, y2, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")

        # gaussian soft target
        y_n = torch.exp(
            -1
            / (self.alpha * 2)
            * (torch.square((grid_x - cx) / w) + torch.square((grid_y - cy) / h))
        )

        cropped_pred = pred["obj"][0, y1:y2, x1:x2]
        cropped_pred = torch.clamp(cropped_pred, min=1e-6, max=1.0 - 1e-6)

        ros_loss = nn.functional.binary_cross_entropy(
            cropped_pred, y_n, reduction="mean"
        )

        # computing the ctr_loss
        # soft-argmax
        weights = cropped_pred / (torch.sum(cropped_pred) + 1e-8)

        pred_ctr = (torch.sum(grid_x * weights), torch.sum(grid_y * weights))

        ctr_loss = (pred_ctr[0] - cx) ** 2 + (pred_ctr[1] - cy) ** 2
        ctr_loss = ctr_loss / (w * h)

        return ros_loss + self.lambda_ctr * ctr_loss, pred_ctr

    def scale_loss(self, pred, tgt_box, pred_ctr):
        cx, cy, w, h = tgt_box
        cx_val = max(0, min(447, round(float(cx))))
        cy_val = max(0, min(447, round(float(cy))))
        cx_p = max(0, min(447, round(float(pred_ctr[0].detach()))))
        cy_p = max(0, min(447, round(float(pred_ctr[1].detach()))))
        half_k = self.k // 2

        y1_pred = max(0, cy_p - half_k)
        y2_pred = min(448, cy_p + half_k + 1)
        x1_pred = max(0, cx_p - half_k)
        x2_pred = min(448, cx_p + half_k + 1)

        cropped_w_pred = pred["w"][0, y1_pred:y2_pred, x1_pred:x2_pred]
        cropped_h_pred = pred["h"][0, y1_pred:y2_pred, x1_pred:x2_pred]

        u = torch.arange(x1_pred, x2_pred, device=self.device)
        v = torch.arange(y1_pred, y2_pred, device=self.device)
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")

        w_q = (w + 2.0 * torch.abs(cx - grid_u)) / 448.0
        h_q = (h + 2.0 * torch.abs(cy - grid_v)) / 448.0

        pred_loss = torch.sum(
            torch.abs(cropped_w_pred - w_q) + torch.abs(cropped_h_pred - h_q)
        )

        y1_tgt = max(0, cy_val - half_k)
        y2_tgt = min(448, cy_val + half_k + 1)
        x1_tgt = max(0, cx_val - half_k)
        x2_tgt = min(448, cx_val + half_k + 1)

        cropped_w_tgt = pred["w"][0, y1_tgt:y2_tgt, x1_tgt:x2_tgt]
        cropped_h_tgt = pred["h"][0, y1_tgt:y2_tgt, x1_tgt:x2_tgt]

        u = torch.arange(x1_tgt, x2_tgt, device=self.device)
        v = torch.arange(y1_tgt, y2_tgt, device=self.device)
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")

        w_q = (w + 2.0 * torch.abs(cx - grid_u)) / 448.0
        h_q = (h + 2.0 * torch.abs(cy - grid_v)) / 448.0

        tgt_loss = torch.sum(
            torch.abs(cropped_w_tgt - w_q) + torch.abs(cropped_h_tgt - h_q)
        )

        pred_count = max(cropped_w_pred.numel(), 1)
        tgt_count = max(cropped_w_tgt.numel(), 1)
        return (pred_loss / pred_count) + (tgt_loss / tgt_count)
