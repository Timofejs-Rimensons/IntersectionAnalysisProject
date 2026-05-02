import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch


class Augmentor:

    def __init__(self, config, mode="train"):
        self.config = config
        img_h, img_w = config.img_size

        base_resize = [
            A.LongestMaxSize(max_size=max(img_h, img_w)),
            A.PadIfNeeded(
                min_height=img_h,
                min_width=img_w,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
        ]

        normalize = [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]

        if mode == "train" and config.use_augmentation:
            augmentations = [
                A.HorizontalFlip(p=0.5),

                A.Affine(
                    scale=(0.9, 1.05),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-1, 1),
                    p=0.3,
                ),

                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        p=0.7
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=5,
                        sat_shift_limit=10,
                        val_shift_limit=10,
                        p=0.7
                    ),
                ], p=0.4),

                A.OneOf([
                    A.GaussianBlur(blur_limit=(1, 3), p=1.0),
                ], p=0.05),

                A.GaussNoise(p=0.05),
            ]

            self.transform = A.Compose(
                base_resize + augmentations + normalize,
                bbox_params=A.BboxParams(
                    format='coco',
                    label_fields=['labels'],
                    min_area=16,
                    min_visibility=0.2
                )
            )
        else:
            self.transform = A.Compose(
                base_resize + normalize,
                bbox_params=A.BboxParams(
                    format='coco',
                    label_fields=['labels'],
                    min_area=1,
                    min_visibility=0.1,
                )
            )

    def __call__(self, image, bboxes, labels):
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            labels=labels
        )
        image = transformed["image"].float()

        boxes = transformed["bboxes"]
        labels = transformed["labels"]

        if len(boxes) > 0:
            boxes_xyxy = []
            for b in boxes:
                x, y, w, h = b
                boxes_xyxy.append([x, y, x + w, y + h])
            boxes_tensor = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }

        return image, target