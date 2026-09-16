import random

import torch
import torchvision.transforms.functional as F
from torch import Generator
from torch.utils.data import ConcatDataset, Dataset, random_split
from torchvision.datasets import VOCDetection


class PascalVOC(Dataset):
    def __init__(self, path, split="train", train_size=0.9, seed=42):
        super().__init__()
        self.split = split
        self.path = path
        self.train_size = train_size
        self.seed = seed

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

        self.target_classes = (
            SEEN_CLASSES if self.split in ["train", "val"] else UNSEEN_CLASSES
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        bboxes = target["annotation"]["object"]

        # if there is only one box
        if not isinstance(bboxes, list):
            bboxes = [bboxes]

        final_bboxes = []
        for box in bboxes:
            if box["name"] in self.target_classes:
                # ignore difficult objects during training
                if self.split == "train" and box.get("difficult", "0") == "1":
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
            # shift by the padding applied to the image
            shifted_xmin = xmin + pad_left
            shifted_xmax = xmax + pad_left
            shifted_ymin = ymin + pad_top
            shifted_ymax = ymax + pad_top

            # convert to normalized [cx, cy, w, h] in [0, 1]
            cx = (shifted_xmin + shifted_xmax) / (2.0 * s)
            cy = (shifted_ymin + shifted_ymax) / (2.0 * s)
            w_box = (shifted_xmax - shifted_xmin) / float(s)
            h_box = (shifted_ymax - shifted_ymin) / float(s)

            norm_boxes.append([cx, cy, w_box, h_box])

        if len(norm_boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes_tensor = torch.tensor(norm_boxes, dtype=torch.float32)

        return tensor_img, boxes_tensor

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = [item[1] for item in batch]
        return images, targets
