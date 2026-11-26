import random
from os.path import join

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import Generator
from torch.utils.data import ConcatDataset, Dataset, random_split
from torchvision.datasets import CocoDetection, VOCDetection

# Official CenterNet (Objects as Points, Zhou et al. 2019) VOC augmentation constants
_CENTERNET_SCALES = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
_CENTERNET_EIG_VAL = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
_CENTERNET_EIG_VEC = np.array(
    [
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938],
    ],
    dtype=np.float32,
)


def _get_border(border: int, size: int) -> int:
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


def _get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return [
        src_point[0] * cs - src_point[1] * sn,
        src_point[0] * sn + src_point[1] * cs,
    ]


def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_affine_transform(
    center,
    scale,
    rot,
    output_size,
    shift=np.array([0, 0], dtype=np.float32),
    inv=0,
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    src_w = scale[0]
    dst_w, dst_h = output_size[0], output_size[1]

    rot_rad = np.pi * rot / 180.0
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    return np.dot(t, new_pt)[:2]


def _color_aug(image: np.ndarray, eig_val: np.ndarray, eig_vec: np.ndarray) -> None:
    """In-place official CenterNet RGB color jitter + PCA lighting noise."""
    gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gs_mean = float(gs.mean())

    ops = ["brightness", "contrast", "saturation"]
    random.shuffle(ops)
    for op in ops:
        alpha = 1.0 + random.uniform(-0.4, 0.4)
        if op == "brightness":
            image *= alpha
        elif op == "contrast":
            image *= alpha
            image += (1.0 - alpha) * gs_mean
        else:
            image *= alpha
            image += (1.0 - alpha) * gs[:, :, None]

    pca_alpha = np.array([random.gauss(0.0, 0.1) for _ in range(3)], dtype=np.float32)
    image += np.dot(eig_vec, eig_val * pca_alpha)


def _apply_centernet_aug(img, raw_boxes, raw_labels):
    """Applies official CenterNet affine crop/flip + color augmentation if enabled."""
    try:
        import run_config as cfg

        apply_aug = getattr(cfg, "CENTERNET_DATA_AUG", True)
        input_size = int(getattr(cfg, "CENTERNET_INPUT_SIZE", 512))
    except Exception:
        apply_aug = True
        input_size = 512

    if not apply_aug:
        return None

    img_np = np.asarray(img.convert("RGB"))
    height, width = img_np.shape[0], img_np.shape[1]

    c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    s = float(max(height, width)) * random.choice(_CENTERNET_SCALES)
    w_border = _get_border(128, width)
    h_border = _get_border(128, height)
    c[0] = float(random.randrange(w_border, width - w_border))
    c[1] = float(random.randrange(h_border, height - h_border))

    flipped = False
    if random.random() < 0.5:
        flipped = True
        img_np = img_np[:, ::-1, :].copy()
        c[0] = width - c[0] - 1.0

    trans_input = _get_affine_transform(c, s, 0.0, [input_size, input_size])
    inp = cv2.warpAffine(
        img_np, trans_input, (input_size, input_size), flags=cv2.INTER_LINEAR
    )
    inp = inp.astype(np.float32) / 255.0
    _color_aug(inp, _CENTERNET_EIG_VAL, _CENTERNET_EIG_VEC)
    tensor_img = torch.from_numpy(np.ascontiguousarray(inp.transpose(2, 0, 1)))

    out_boxes = []
    out_labels = []
    max_coord = float(input_size - 1)
    for box, label in zip(raw_boxes, raw_labels):
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        if flipped:
            x1, x2 = width - x2 - 1.0, width - x1 - 1.0
        p1 = _affine_transform((x1, y1), trans_input)
        p2 = _affine_transform((x2, y2), trans_input)
        bx1 = float(np.clip(p1[0], 0.0, max_coord))
        by1 = float(np.clip(p1[1], 0.0, max_coord))
        bx2 = float(np.clip(p2[0], 0.0, max_coord))
        by2 = float(np.clip(p2[1], 0.0, max_coord))
        if bx2 > bx1 and by2 > by1:
            out_boxes.append([bx1, by1, bx2, by2])
            out_labels.append(int(label))

    if len(out_boxes) > 0:
        boxes_t = torch.tensor(out_boxes, dtype=torch.float32)
        labels_t = torch.tensor(out_labels, dtype=torch.int64)
    else:
        boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.zeros((0,), dtype=torch.int64)

    return tensor_img, {"boxes": boxes_t, "labels": labels_t}


VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
VOC_CLASS_TO_IDX = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}


class PascalVOC(Dataset):
    def __init__(
        self,
        path,
        split="train",
        train_size=0.9,
        seed=42,
        all_classes=None,
        model_type=None,
    ):
        super().__init__()
        self.split = split
        self.path = path
        self.train_size = train_size
        self.seed = seed

        if all_classes is None:
            try:
                import run_config as cfg

                all_classes = getattr(cfg, "ALL_CLASSES", False)
            except Exception:
                all_classes = False
        self.all_classes = all_classes

        if model_type is None:
            try:
                import run_config as cfg

                model_type = getattr(cfg, "MODEL_NAME", "bfor")
            except Exception:
                model_type = "bfor"
        self.model_type = model_type

        if self.split in ["train", "val"]:
            voc2007_train = VOCDetection(
                root=self.path, year="2007", image_set="trainval", download=False
            )
            voc2012_train = VOCDetection(
                root=self.path, year="2012", image_set="trainval", download=False
            )

            full_set = ConcatDataset([voc2007_train, voc2012_train])

            if self.train_size >= 1.0:
                self.dataset = full_set if self.split == "train" else []
            else:
                gen = Generator().manual_seed(self.seed)

                train_set, val_set = random_split(
                    full_set,
                    lengths=[self.train_size, 1.0 - self.train_size],
                    generator=gen,
                )

                if self.split == "train":
                    self.dataset = train_set
                else:
                    self.dataset = val_set

        elif self.split == "test":
            self.dataset = VOCDetection(
                root=self.path, year="2007", image_set="test", download=False
            )
        else:
            raise ValueError("split has to be 'train', 'val' or 'test'")

        SEEN_CLASSES = {
            "aeroplane",
            "bicycle",
            "bird",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
        }
        UNSEEN_CLASSES = {"boat", "cow", "tvmonitor"}
        ALL_CLASSES = SEEN_CLASSES | UNSEEN_CLASSES

        if self.all_classes:
            self.target_classes = ALL_CLASSES
        else:
            self.target_classes = (
                SEEN_CLASSES if self.split in ["train", "val"] else UNSEEN_CLASSES
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        bboxes = target["annotation"]["object"]
        labels = []

        # if there is only one box
        if not isinstance(bboxes, list):
            bboxes = [bboxes]

        # Mode FCOS / CenterNet: native resolution, xyxy pixel boxes and integer class IDs
        if self.model_type in ["fcos", "bfor_fcos", "centernet"]:
            fcos_boxes = []
            fcos_labels = []
            for box in bboxes:
                if box["name"] in self.target_classes:
                    if self.split in ["train", "val"] and box.get("difficult", "0") == "1":
                        continue
                    b = box["bndbox"]
                    x1 = float(b["xmin"])
                    y1 = float(b["ymin"])
                    x2 = float(b["xmax"])
                    y2 = float(b["ymax"])
                    if x2 > x1 and y2 > y1:
                        fcos_boxes.append([x1, y1, x2, y2])
                        fcos_labels.append(VOC_CLASS_TO_IDX[box["name"]])
                        labels.append(box["name"])

            if self.model_type == "centernet" and self.split == "train":
                aug_res = _apply_centernet_aug(img, fcos_boxes, fcos_labels)
                if aug_res is not None:
                    return aug_res

            tensor_img = F.to_tensor(img)
            if len(fcos_boxes) == 0:
                target_dict = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
            else:
                target_dict = {
                    "boxes": torch.tensor(fcos_boxes, dtype=torch.float32),
                    "labels": torch.tensor(fcos_labels, dtype=torch.int64),
                }

            if self.split == "test":
                return tensor_img, target_dict, labels

            return tensor_img, target_dict

        # Mode B-FOR: 448x448 padded canvas and normalized cxcywh boxes
        final_bboxes = []
        for box in bboxes:
            if box["name"] in self.target_classes:
                if box.get("difficult", "0") == "1":
                    continue

                b = box["bndbox"]
                final_bboxes.append(
                    [
                        float(b["xmin"]),
                        float(b["ymin"]),
                        float(b["xmax"]),
                        float(b["ymax"]),
                    ]
                )
                labels.append(box["name"])

        # max 10 boxes per image during training
        if self.split == "train" and len(final_bboxes) > 10:
            final_bboxes = random.sample(final_bboxes, k=10)

        w, h = img.size
        s = max(w, h)

        pad_left = (s - w) // 2
        pad_right = s - w - pad_left
        pad_top = (s - h) // 2
        pad_bottom = s - h - pad_top

        padded_img = F.pad(
            img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0
        )
        resized_img = F.resize(padded_img, [448, 448])
        tensor_img = F.to_tensor(resized_img)

        norm_boxes = []
        for xmin, ymin, xmax, ymax in final_bboxes:
            shifted_xmin = xmin + pad_left
            shifted_xmax = xmax + pad_left
            shifted_ymin = ymin + pad_top
            shifted_ymax = ymax + pad_top

            cx = (shifted_xmin + shifted_xmax) / (2.0 * s)
            cy = (shifted_ymin + shifted_ymax) / (2.0 * s)
            w_box = (shifted_xmax - shifted_xmin) / float(s)
            h_box = (shifted_ymax - shifted_ymin) / float(s)

            norm_boxes.append([cx, cy, w_box, h_box])

        if len(norm_boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes_tensor = torch.tensor(norm_boxes, dtype=torch.float32)

        if self.split == "test":
            return tensor_img, boxes_tensor, labels

        return tensor_img, boxes_tensor

    @staticmethod
    def collate_fn(batch):
        if len(batch) > 0 and isinstance(batch[0][1], dict):
            images = tuple(item[0] for item in batch)
            targets = tuple(item[1] for item in batch)
            return images, targets
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = [item[1] for item in batch]
        return images, targets


class COCO2017(Dataset):
    def __init__(
        self,
        path,
        split="train",
        train_size=0.9,
        seed=42,
        all_classes=None,
        model_type=None,
    ):
        super().__init__()
        self.split = split
        self.path = path
        self.train_size = train_size
        self.seed = seed

        if all_classes is None:
            try:
                import run_config as cfg

                all_classes = getattr(cfg, "ALL_CLASSES", False)
            except Exception:
                all_classes = False
        self.all_classes = all_classes

        if model_type is None:
            try:
                import run_config as cfg

                model_type = getattr(cfg, "MODEL_NAME", "bfor")
            except Exception:
                model_type = "bfor"
        self.model_type = model_type

        if self.split in ["train", "val"]:
            coco_train = CocoDetection(
                root=join(self.path, "coco17", "train2017"),
                annFile=join(
                    self.path, "coco17", "annotations", "instances_train2017.json"
                ),
            )

            self.coco = coco_train.coco

            if self.train_size >= 1.0:
                self.dataset = coco_train if self.split == "train" else []
            else:
                gen = Generator().manual_seed(self.seed)

                train_set, val_set = random_split(
                    coco_train,
                    lengths=[self.train_size, 1.0 - self.train_size],
                    generator=gen,
                )

                if self.split == "train":
                    self.dataset = train_set
                else:
                    self.dataset = val_set

        elif self.split == "test":
            self.dataset = CocoDetection(
                root=join(self.path, "coco17", "val2017"),
                annFile=join(
                    self.path, "coco17", "annotations", "instances_val2017.json"
                ),
            )
            self.coco = self.dataset.coco
        else:
            raise ValueError("split has to be 'train', 'val' or 'test'")

        ALL_CLASSES = {cls["name"] for cls in self.coco.cats.values()}

        VOC_CLASSES_SET = {
            "airplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "couch",
            "cow",
            "dining table",
            "dog",
            "horse",
            "motorcycle",
            "person",
            "potted plant",
            "sheep",
            "train",
            "tv",
        }

        if self.all_classes:
            self.target_classes = ALL_CLASSES
        else:
            self.target_classes = ALL_CLASSES - VOC_CLASSES_SET

        self.target_cat_ids = {
            cat_id: cat["name"] for cat_id, cat in self.coco.cats.items()
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        target = [
            (tgt["bbox"], self.target_cat_ids[tgt["category_id"]])
            for tgt in target
            if tgt["iscrowd"] == 0
        ]

        # Mode FCOS / CenterNet: native resolution, xyxy pixel boxes and integer class IDs
        if self.model_type in ["fcos", "bfor_fcos", "centernet"]:
            fcos_boxes = []
            fcos_labels = []
            labels = []
            for tgt in target:
                cat_name = tgt[1]
                if cat_name in self.target_classes:
                    x, y, w_box, h_box = tgt[0]
                    x1 = float(x)
                    y1 = float(y)
                    x2 = x1 + float(w_box)
                    y2 = y1 + float(h_box)
                    if x2 > x1 and y2 > y1:
                        fcos_boxes.append([x1, y1, x2, y2])
                        fcos_labels.append(VOC_CLASS_TO_IDX.get(cat_name, 1))
                        labels.append(cat_name)

            if self.model_type == "centernet" and self.split == "train":
                aug_res = _apply_centernet_aug(img, fcos_boxes, fcos_labels)
                if aug_res is not None:
                    return aug_res

            tensor_img = F.to_tensor(img)
            if len(fcos_boxes) == 0:
                target_dict = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
            else:
                target_dict = {
                    "boxes": torch.tensor(fcos_boxes, dtype=torch.float32),
                    "labels": torch.tensor(fcos_labels, dtype=torch.int64),
                }

            if self.split == "test":
                return tensor_img, target_dict, labels

            return tensor_img, target_dict

        # Mode B-FOR: 448x448 padded canvas and normalized cxcywh boxes
        final_target = [tgt[0] for tgt in target if tgt[1] in self.target_classes]
        labels = [tgt[1] for tgt in target if tgt[1] in self.target_classes]

        w, h = img.size
        s = max(w, h)

        pad_left = (s - w) // 2
        pad_right = s - w - pad_left
        pad_top = (s - h) // 2
        pad_bottom = s - h - pad_top

        padded_img = F.pad(
            img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0
        )
        resized_img = F.resize(padded_img, [448, 448])
        tensor_img = F.to_tensor(resized_img)

        norm_boxes = []
        for x, y, bw, bh in final_target:
            cx = (x + pad_left + bw / 2.0) / float(s)
            cy = (y + pad_top + bh / 2.0) / float(s)
            w_norm = bw / float(s)
            h_norm = bh / float(s)

            norm_boxes.append([cx, cy, w_norm, h_norm])

        if len(norm_boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes_tensor = torch.tensor(norm_boxes, dtype=torch.float32)

        if self.split == "test":
            return tensor_img, boxes_tensor, labels

        return tensor_img, boxes_tensor

    @staticmethod
    def collate_fn(batch):
        if len(batch) > 0 and isinstance(batch[0][1], dict):
            images = tuple(item[0] for item in batch)
            targets = tuple(item[1] for item in batch)
            return images, targets
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = [item[1] for item in batch]
        return images, targets
