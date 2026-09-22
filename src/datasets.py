import math
import random
import xml.etree.ElementTree as ET

import torch
import torchvision.transforms.functional as F
from torch import Generator
from torch.utils.data import ConcatDataset, Dataset, random_split, Sampler
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
        labels = []

        # if there is only one box
        if not isinstance(bboxes, list):
            bboxes = [bboxes]

        final_bboxes = []
        for box in bboxes:
            if box["name"] in self.target_classes:
                # ignore difficult objects during training
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

        if self.split == "test":
            meta = {
                "orig_w": w,
                "orig_h": h,
                "s": s,
                "pad_left": pad_left,
                "pad_top": pad_top,
                "orig_boxes": (
                    torch.tensor(final_bboxes, dtype=torch.float32)
                    if len(final_bboxes) > 0
                    else torch.zeros((0, 4), dtype=torch.float32)
                ),
            }
            return tensor_img, boxes_tensor, labels, meta

        return tensor_img, boxes_tensor

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = [item[1] for item in batch]
        if len(batch[0]) > 2:
            labels = [item[2] for item in batch]
            if len(batch[0]) > 3:
                metas = [item[3] for item in batch]
                return images, targets, labels, metas
            return images, targets, labels
        return images, targets


def get_scale_indices(dataset, sml_cutoff=64.0, lrg_cutoff=96.0):
    """
    Scans the dataset annotations to group image indices into small (P3, < sml_cutoff),
    medium (P4, sml_cutoff <= size < lrg_cutoff), and large (P5, >= lrg_cutoff).
    Uses fast direct XML parsing over underlying VOCDetection datasets (~0.4s).
    """
    p3, p4, p5 = [], [], []
    target_classes = getattr(dataset, "target_classes", None)

    underlying = getattr(dataset, "dataset", dataset)
    if hasattr(underlying, "indices") and hasattr(underlying, "dataset"):
        subset_indices = underlying.indices
        concat_or_voc = underlying.dataset
        if hasattr(concat_or_voc, "datasets"):
            all_xmls = []
            for d in concat_or_voc.datasets:
                all_xmls.extend(d.annotations)
        elif hasattr(concat_or_voc, "annotations"):
            all_xmls = concat_or_voc.annotations
        else:
            all_xmls = None
    elif hasattr(underlying, "annotations"):
        subset_indices = list(range(len(underlying)))
        all_xmls = underlying.annotations
    else:
        all_xmls = None

    if all_xmls is not None and target_classes is not None:
        for sub_idx, full_idx in enumerate(subset_indices):
            xml_path = all_xmls[full_idx]
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size_elem = root.find("size")
            w = float(size_elem.find("width").text)
            h = float(size_elem.find("height").text)
            s = max(w, h)
            scale = 448.0 / s

            has_p3 = has_p4 = has_p5 = False
            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                if cls_name not in target_classes:
                    continue
                diff = obj.find("difficult")
                if diff is not None and diff.text == "1":
                    continue
                b = obj.find("bndbox")
                bw = (float(b.find("xmax").text) - float(b.find("xmin").text)) * scale
                bh = (float(b.find("ymax").text) - float(b.find("ymin").text)) * scale
                box_size = math.sqrt(max(bw, 0.0) * max(bh, 0.0))
                if box_size < sml_cutoff:
                    has_p3 = True
                elif box_size < lrg_cutoff:
                    has_p4 = True
                else:
                    has_p5 = True

            if has_p3:
                p3.append(sub_idx)
            if has_p4:
                p4.append(sub_idx)
            if has_p5:
                p5.append(sub_idx)
    else:
        # Fallback: inspect items directly
        for idx in range(len(dataset)):
            _, boxes = dataset[idx]
            if len(boxes) == 0:
                continue
            sizes = torch.sqrt(boxes[:, 2] * boxes[:, 3]) * 448.0
            if (sizes < sml_cutoff).any():
                p3.append(idx)
            if ((sizes >= sml_cutoff) & (sizes < lrg_cutoff)).any():
                p4.append(idx)
            if (sizes >= lrg_cutoff).any():
                p5.append(idx)

    return p3, p4, p5


class StratifiedBatchSampler(Sampler):
    """
    Batch sampler that guarantees each training batch contains at least one sample
    with a small object (P3, < 64px), medium object (P4, 64-96px), and large object
    (P5, >= 96px), exactly implementing the authors' coco_data_generator_stratified.
    """

    def __init__(
        self,
        dataset,
        batch_size=4,
        sml_cutoff=64.0,
        lrg_cutoff=96.0,
        require_levels=("P3", "P4", "P5"),
        min_per_level=1,
        seed=42,
        drop_last=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.sml_cutoff = sml_cutoff
        self.lrg_cutoff = lrg_cutoff
        self.require_levels = tuple(require_levels)
        self.min_per_level = min_per_level
        self.seed = seed
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        # Precompute scale pools
        p3, p4, p5 = get_scale_indices(
            dataset, sml_cutoff=sml_cutoff, lrg_cutoff=lrg_cutoff
        )
        self.pools = {"P3": p3, "P4": p4, "P5": p5}

        for lvl in self.require_levels:
            if len(self.pools[lvl]) == 0:
                raise RuntimeError(
                    f"No images found containing scale level {lvl}. Cannot stratify."
                )

        self.dataset_size = len(dataset)
        if self.drop_last:
            self.num_batches = self.dataset_size // self.batch_size
        else:
            self.num_batches = (
                self.dataset_size + self.batch_size - 1
            ) // self.batch_size

    def __len__(self):
        return self.num_batches

    def _pick_next(self, pool, ptr, chosen):
        """Picks the next element from pool, ensuring it is not already in chosen."""
        for _ in range(len(pool)):
            if ptr >= len(pool):
                self.rng.shuffle(pool)
                ptr = 0
            item = pool[ptr]
            ptr += 1
            if item not in chosen:
                return item, ptr
        return pool[ptr % len(pool)], ptr

    def __iter__(self):
        # Active pools shuffled for the epoch
        active_pools = {lvl: list(self.pools[lvl]) for lvl in self.require_levels}
        for lvl in self.require_levels:
            self.rng.shuffle(active_pools[lvl])

        all_pool = list(range(self.dataset_size))
        self.rng.shuffle(all_pool)

        ptrs = {lvl: 0 for lvl in self.require_levels}
        ptr_all = 0

        for _ in range(self.num_batches):
            batch = []
            chosen_set = set()

            # 1. Guarantee min_per_level from required scale levels
            if self.batch_size >= len(self.require_levels) * self.min_per_level:
                for lvl in self.require_levels:
                    for _ in range(self.min_per_level):
                        item, ptrs[lvl] = self._pick_next(
                            active_pools[lvl], ptrs[lvl], chosen_set
                        )
                        batch.append(item)
                        chosen_set.add(item)

            # 2. Fill remaining slots with random samples from all_pool
            while len(batch) < self.batch_size:
                item, ptr_all = self._pick_next(all_pool, ptr_all, chosen_set)
                batch.append(item)
                chosen_set.add(item)

            yield batch
