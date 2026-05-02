from PIL import Image
import numpy as np
import os
import json
import random

from torch.utils.data import Dataset, DataLoader
from services.Augmentor import Augmentor

class CocoDetectionDataset(Dataset):

    def __init__(self, img_dir, ann_json, config, mode="train"):
        self.img_dir = img_dir
        self.config = config
        self.mode = mode

        with open(ann_json, 'r') as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        self.img_ids = []
        self.img_anns = {}

        anns_by_img = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in anns_by_img:
                anns_by_img[img_id] = []
            anns_by_img[img_id].append(ann)

        for img_id, anns in anns_by_img.items():
            valid_anns = [
                a for a in anns
                if a["bbox"][2] > 0 and a["bbox"][3] > 0
            ]
            if valid_anns:
                self.img_ids.append(img_id)
                self.img_anns[img_id] = valid_anns

        self.augmentor = Augmentor(config, mode=mode)

        print(f"[{mode.upper()}] Loaded {len(self.img_ids)} images "
              f"with {sum(len(v) for v in self.img_anns.values())} annotations")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        anns = self.img_anns[img_id]

        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = np.array(Image.open(img_path).convert("RGB"))

        bboxes = [ann["bbox"] for ann in anns]
        labels = [ann["category_id"] for ann in anns]

        image, target = self.augmentor(image, bboxes, labels)

        target["image_id"] = img_id

        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class EpochSubsetDataset(Dataset):

    def __init__(self, full_dataset, epoch_size):
        self.full_dataset = full_dataset
        self.epoch_size = min(epoch_size, len(full_dataset))
        self.indices = random.sample(range(len(self.full_dataset)), self.epoch_size)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        return self.full_dataset[self.indices[idx]]


def build_dataloaders(config):

    train_dataset = CocoDetectionDataset(
        img_dir=os.path.join(config.data_dir, config.train_img_dir),
        ann_json=os.path.join(config.data_dir, config.train_json),
        config=config,
        mode="train"
    )

    val_dataset = CocoDetectionDataset(
        img_dir=os.path.join(config.data_dir, config.val_img_dir),
        ann_json=os.path.join(config.data_dir, config.val_json),
        config=config,
        mode="val"
    )

    epoch_size = getattr(config, 'epoch_size', len(train_dataset))
    val_epoch_size = getattr(config, 'val_epoch_size', len(val_dataset))

    train_subset = EpochSubsetDataset(train_dataset, epoch_size)
    val_subset = EpochSubsetDataset(val_dataset, val_epoch_size)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader