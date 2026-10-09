import random
from os.path import join

import torch
import torchvision.transforms.functional as F
from torch import Generator
from torch.utils.data import ConcatDataset, Dataset, random_split
from torchvision.datasets import CocoDetection, VOCDetection


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
            tensor_img = F.to_tensor(img)
            fcos_boxes = []
            fcos_labels = []
            for box in bboxes:
                if box["name"] in self.target_classes:
                    if self.split == "train" and box.get("difficult", "0") == "1":
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
            tensor_img = F.to_tensor(img)
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
