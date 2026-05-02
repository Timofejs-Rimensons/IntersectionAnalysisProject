import os
import json
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random as rnd
from collections import defaultdict


class VisDroneMOTCocoConvertor:
    def __init__(self):
        self.visdrone_mot_cat_map = {
            1:  {"id": 0, "name": "person"},
            3:  {"id": 1, "name": "bicycle"},
            4:  {"id": 2, "name": "car"},
            5:  {"id": 2, "name": "car"},
            6:  {"id": 3, "name": "truck"},
            7:  {"id": 1, "name": "bicycle"},
            8:  {"id": 1, "name": "bicycle"},
            9:  {"id": 4, "name": "bus"},
            10: {"id": 1, "name": "bicycle"},
        }

        self.default_area_percentile_thresholds = {
            "person":  15.0,
            "bicycle": 15.0,
            "car":     15.0,
            "truck":   0.0,
            "bus":     0.0,
        }


    @staticmethod
    def _compute_per_class_area_thresholds(
        annotations: list,
        categories: list,
        percentile_thresholds: dict,
    ) -> dict:
        cat_id_to_name = {c["id"]: c["name"] for c in categories}
        cat_name_to_id = {c["name"]: c["id"] for c in categories}

        areas_by_cat: dict[int, list[float]] = defaultdict(list)
        for ann in annotations:
            cid = ann["category_id"]
            area = ann.get("area")
            if area is None:
                bx, by, bw, bh = ann["bbox"]
                area = bw * bh
            areas_by_cat[cid].append(area)

        thresholds: dict[int, float] = {}

        for cat in categories:
            cid = cat["id"]
            cname = cat["name"]
            pct = percentile_thresholds.get(cname, 0.0)

            if pct <= 0 or cid not in areas_by_cat or len(areas_by_cat[cid]) == 0:
                thresholds[cid] = 0.0
                continue

            areas = np.array(areas_by_cat[cid])
            threshold_area = float(np.percentile(areas, pct))
            thresholds[cid] = threshold_area

        return thresholds

    @staticmethod
    def _filter_annotations_by_area(
        annotations: list,
        area_thresholds: dict,
    ) -> tuple[list, dict]:
        filtered = []
        removal_stats: dict[int, int] = defaultdict(int)

        for ann in annotations:
            cid = ann["category_id"]
            area = ann.get("area")
            if area is None:
                bx, by, bw, bh = ann["bbox"]
                area = bw * bh

            min_area = area_thresholds.get(cid, 0.0)
            if area < min_area:
                removal_stats[cid] += 1
                continue

            filtered.append(ann)

        return filtered, dict(removal_stats)

    def _print_filter_summary(
        self,
        categories: list,
        area_thresholds: dict,
        percentile_thresholds: dict,
        removal_stats: dict,
        total_before: int,
        total_after: int,
    ):
        cat_id_to_name = {c["id"]: c["name"] for c in categories}

        print("\n" + "─" * 60)
        print("Small Bounding-Box Filtering (per-class percentile)")
        print("─" * 60)
        print(f"  {'Category':<15s} {'Percentile':>10s} {'Min Area (px²)':>15s} {'Removed':>10s}")
        print(f"  {'─'*15:<15s} {'─'*10:>10s} {'─'*15:>15s} {'─'*10:>10s}")

        for cat in categories:
            cid = cat["id"]
            cname = cat["name"]
            pct = percentile_thresholds.get(cname, 0.0)
            min_area = area_thresholds.get(cid, 0.0)
            removed = removal_stats.get(cid, 0)
            print(f"  {cname:<15s} {pct:>9.1f}% {min_area:>15.1f} {removed:>10,d}")

        print(f"\n  Total annotations before: {total_before:,}")
        print(f"  Total annotations after:  {total_after:,}")
        print(f"  Total removed:            {total_before - total_after:,}")


    def visdrone_mot_to_coco_detection(
        self,
        seq_dir: str,
        ann_dir: str,
        output_json: str,
        ignore_dir: str = None,
        area_percentile_thresholds: dict = None,
    ) -> dict:

        tracking_coco = self.visdrone_mot_to_coco_tracking(
            seq_dir=seq_dir,
            ann_dir=ann_dir,
            output_json=None,
            ignore_dir=ignore_dir,
            area_percentile_thresholds=area_percentile_thresholds,
        )
        detection_coco = {
            "info": tracking_coco["info"],
            "licenses": tracking_coco["licenses"],
            "images": [],
            "annotations": [],
            "categories": tracking_coco["categories"],
        }

        image_id_map = {}
        new_image_id = 1
        new_annotation_id = 1

        for img in tracking_coco["images"]:
            image_id_map[img["id"]] = new_image_id

            detection_coco["images"].append({
                "id":        new_image_id,
                "file_name": img["file_name"],
                "width":     img["width"],
                "height":    img["height"],
            })
            new_image_id += 1

        for ann in tracking_coco["annotations"]:
            old_image_id = ann["image_id"]
            new_iid = image_id_map[old_image_id]

            detection_coco["annotations"].append({
                "id":          new_annotation_id,
                "image_id":    new_iid,
                "category_id": ann["category_id"],
                "bbox":        ann["bbox"],
                "area":        ann["area"],
                "iscrowd":     ann["iscrowd"],
            })
            new_annotation_id += 1

        os.makedirs(
            os.path.dirname(output_json) or '.', exist_ok=True
        )
        with open(output_json, 'w') as f:
            json.dump(detection_coco, f, indent=2)

        print("\n" + "=" * 60)
        print("VisDrone MOT → COCO Detection Format: Conversion Complete!")
        print("=" * 60)
        print(f"  Images:       {len(detection_coco['images'])}")
        print(f"  Annotations:  {len(detection_coco['annotations'])}")
        print(f"  Categories:   {len(detection_coco['categories'])}")
        print(f"  Saved to:     {output_json}")

        print("\nPer-category annotation counts:")
        for cat in detection_coco["categories"]:
            count = sum(
                1 for a in detection_coco["annotations"]
                if a["category_id"] == cat["id"]
            )
            print(f"   {cat['name']:20s}: {count:,}")

        return detection_coco

    def visdrone_mot_to_coco_tracking(
        self,
        seq_dir: str,
        ann_dir: str,
        output_json: str,
        ignore_dir: str = None,
        area_percentile_thresholds: dict = None,
    ) -> dict:

        if area_percentile_thresholds is None:
            pct_thresholds = dict(self.default_area_percentile_thresholds)
        else:
            pct_thresholds = dict(area_percentile_thresholds)

        cat_source = self.visdrone_mot_cat_map

        seen_cats = {}
        for _, cat_info in cat_source.items():
            cid = cat_info["id"]
            if cid not in seen_cats:
                seen_cats[cid] = {
                    "id": cid,
                    "name": cat_info["name"],
                    "supercategory": "none",
                }
        categories = sorted(seen_cats.values(), key=lambda x: x["id"])

        coco = {
            "info": {
                "description": (
                    "VisDrone MOT converted to COCO tracking format. "
                    "Source: VisDrone2019-MOT"
                ),
                "version": "1.0",
                "year": 2024,
                "source_format": "VisDrone MOT",
            },
            "licenses": [],
            "videos": [],
            "images": [],
            "annotations": [],
            "tracks": [],
            "categories": categories,
        }

        sequences = sorted([
            d for d in os.listdir(seq_dir)
            if os.path.isdir(os.path.join(seq_dir, d))
        ])

        if not sequences:
            raise FileNotFoundError(
                f"No sequence folders found in {seq_dir}"
            )

        print(f"Found {len(sequences)} video sequences")
        print(f"Categories: {[c['name'] for c in categories]}")

        video_id = 0
        image_id = 0
        annotation_id = 0
        global_track_id = 0
        track_map = {}

        stats = {
            "skipped_sequences":   0,
            "skipped_annotations": 0,
            "ignored_by_score":    0,
            "ignored_by_category": 0,
            "total_frames":        0,
            "total_annotations":   0,
        }

        for seq_name in tqdm(sequences, desc="Converting VisDrone MOT"):
            seq_path = os.path.join(seq_dir, seq_name)
            ann_path = os.path.join(ann_dir, seq_name + ".txt")

            if not os.path.exists(ann_path):
                print(f"    Missing annotation: {seq_name}.txt")
                stats["skipped_sequences"] += 1
                continue

            frame_files = sorted([
                f for f in os.listdir(seq_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

            if not frame_files:
                print(f"    No images in: {seq_name}")
                stats["skipped_sequences"] += 1
                continue

            try:
                first_img = Image.open(
                    os.path.join(seq_path, frame_files[0])
                )
                img_w, img_h = first_img.size
            except Exception as e:
                print(f"    Cannot read first frame of {seq_name}: {e}")
                stats["skipped_sequences"] += 1
                continue

            video_id += 1

            coco["videos"].append({
                "id":       video_id,
                "name":     seq_name,
                "width":    img_w,
                "height":   img_h,
                "n_frames": len(frame_files),
            })

            frame_annotations = {}

            with open(ann_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(',')

                    if len(parts) < 10:
                        stats["skipped_annotations"] += 1
                        continue

                    try:
                        frame_idx  = int(parts[0])
                        target_id  = int(parts[1])
                        bbox_left  = float(parts[2])
                        bbox_top   = float(parts[3])
                        bbox_w     = float(parts[4])
                        bbox_h     = float(parts[5])
                        score      = int(parts[6])
                        obj_cat_vd = int(parts[7])
                        truncation = int(parts[8])
                        occlusion  = int(parts[9])
                    except (ValueError, IndexError):
                        stats["skipped_annotations"] += 1
                        continue

                    if obj_cat_vd == 0:
                        stats["ignored_by_category"] += 1
                        continue

                    if score == 0:
                        stats["ignored_by_score"] += 1
                        continue

                    if obj_cat_vd not in cat_source:
                        stats["skipped_annotations"] += 1
                        continue

                    if bbox_w <= 0 or bbox_h <= 0:
                        stats["skipped_annotations"] += 1
                        continue

                    if frame_idx not in frame_annotations:
                        frame_annotations[frame_idx] = []

                    frame_annotations[frame_idx].append({
                        "target_id":   target_id,
                        "bbox":        [bbox_left, bbox_top, bbox_w, bbox_h],
                        "obj_cat_vd":  obj_cat_vd,
                        "truncation":  truncation,
                        "occlusion":   occlusion,
                    })

            frame_name_to_num = {}
            for fname in frame_files:
                stem = os.path.splitext(fname)[0]
                try:
                    num = int(stem)
                except ValueError:
                    num = frame_files.index(fname) + 1
                frame_name_to_num[fname] = num

            for frame_name in frame_files:
                frame_num = frame_name_to_num[frame_name]
                image_id += 1
                stats["total_frames"] += 1

                coco["images"].append({
                    "id":        image_id,
                    "file_name": os.path.join(seq_name, frame_name),
                    "video_id":  video_id,
                    "frame_id":  frame_num,
                    "width":     img_w,
                    "height":    img_h,
                })

                if frame_num not in frame_annotations:
                    continue

                for ann in frame_annotations[frame_num]:
                    local_key = (video_id, ann["target_id"])

                    if local_key not in track_map:
                        global_track_id += 1
                        track_map[local_key] = global_track_id

                        track_cat = cat_source[ann["obj_cat_vd"]]["id"]

                        coco["tracks"].append({
                            "id":          global_track_id,
                            "video_id":    video_id,
                            "category_id": track_cat,
                        })

                    g_track_id = track_map[local_key]

                    coco_cat_id = cat_source[ann["obj_cat_vd"]]["id"]

                    bx, by, bw, bh = ann["bbox"]
                    bx = max(0, min(bx, img_w))
                    by = max(0, min(by, img_h))
                    bw = min(bw, img_w - bx)
                    bh = min(bh, img_h - by)

                    if bw <= 0 or bh <= 0:
                        stats["skipped_annotations"] += 1
                        continue

                    annotation_id += 1
                    stats["total_annotations"] += 1

                    coco["annotations"].append({
                        "id":          annotation_id,
                        "image_id":    image_id,
                        "category_id": coco_cat_id,
                        "track_id":    g_track_id,
                        "bbox":        [bx, by, bw, bh],
                        "area":        bw * bh,
                        "iscrowd":     0,
                        "truncation":  ann["truncation"],
                        "occlusion":   ann["occlusion"],
                    })

        total_before_filter = len(coco["annotations"])

        if pct_thresholds and any(v > 0 for v in pct_thresholds.values()):
            abs_thresholds = self._compute_per_class_area_thresholds(
                annotations=coco["annotations"],
                categories=categories,
                percentile_thresholds=pct_thresholds,
            )

            coco["annotations"], removal_stats = self._filter_annotations_by_area(
                annotations=coco["annotations"],
                area_thresholds=abs_thresholds,
            )

            total_after_filter = len(coco["annotations"])

            self._print_filter_summary(
                categories=categories,
                area_thresholds=abs_thresholds,
                percentile_thresholds=pct_thresholds,
                removal_stats=removal_stats,
                total_before=total_before_filter,
                total_after=total_after_filter,
            )

            stats["total_annotations"] = total_after_filter

            surviving_track_ids = {a["track_id"] for a in coco["annotations"]}
            tracks_before = len(coco["tracks"])
            coco["tracks"] = [
                t for t in coco["tracks"]
                if t["id"] in surviving_track_ids
            ]
            tracks_removed = tracks_before - len(coco["tracks"])
            if tracks_removed > 0:
                print(f"  Pruned {tracks_removed} orphaned tracks "
                      f"(no remaining annotations)")

            coco["info"]["bbox_filter"] = {
                "method": "per_class_area_percentile",
                "percentile_thresholds": pct_thresholds,
                "absolute_area_thresholds": {
                    cat_id_name: abs_thresholds.get(cid, 0.0)
                    for cid, cat_id_name in (
                        (c["id"], c["name"]) for c in categories
                    )
                },
                "annotations_removed": total_before_filter - total_after_filter,
                "tracks_pruned": tracks_removed,
            }
        else:
            print("\n  Small-bbox filtering: DISABLED (all percentiles ≤ 0)")

        if output_json is not None:
            os.makedirs(
                os.path.dirname(output_json) or '.', exist_ok=True
            )
            with open(output_json, 'w') as f:
                json.dump(coco, f, indent=2)

        print("\n" + "=" * 60)
        print("VisDrone MOT → COCO Tracking: Conversion Complete!")
        print("=" * 60)
        print(f"  Videos:       {len(coco['videos'])}")
        print(f"  Frames:       {stats['total_frames']}")
        print(f"  Annotations:  {stats['total_annotations']}")
        print(f"  Tracks:       {len(coco['tracks'])}")
        print(f"  Categories:   {len(coco['categories'])}")
        print(f"  Skipped seq:  {stats['skipped_sequences']}")
        print(f"  Skipped ann:  {stats['skipped_annotations']}")
        print(f"  Ignored (score=0):    {stats['ignored_by_score']}")
        print(f"  Ignored (category=0): {stats['ignored_by_category']}")
        if output_json:
            print(f"  Saved to:     {output_json}")

        print("\nPer-category annotation counts:")
        for cat in categories:
            count = sum(
                1 for a in coco["annotations"]
                if a["category_id"] == cat["id"]
            )
            print(f"   {cat['name']:20s}: {count:,}")

        print(f"\nPer-category track counts:")
        for cat in categories:
            count = sum(
                1 for t in coco["tracks"]
                if t["category_id"] == cat["id"]
            )
            print(f"   {cat['name']:20s}: {count:,}")

        print(f"\nPer-video summary:")
        imgs_by_vid = {}
        for img in coco["images"]:
            vid = img["video_id"]
            if vid not in imgs_by_vid:
                imgs_by_vid[vid] = set()
            imgs_by_vid[vid].add(img["id"])

        for video in coco["videos"]:
            vid = video["id"]
            n_frames = len(imgs_by_vid.get(vid, set()))
            img_ids = imgs_by_vid.get(vid, set())
            n_anns = sum(
                1 for a in coco["annotations"]
                if a["image_id"] in img_ids
            )
            n_tracks = sum(
                1 for t in coco["tracks"]
                if t["video_id"] == vid
            )
            print(f"   {video['name']:35s}: "
                  f"{n_frames:5d} frames, "
                  f"{n_tracks:4d} tracks, "
                  f"{n_anns:6d} detections")

        return coco

    def visdrone_mot_to_coco_train_val(
        self,
        seq_dir: str,
        ann_dir: str,
        output_dir: str,
        val_split: float = 0.2,
        area_percentile_thresholds: dict = None,
    ):
        full_coco = self.visdrone_mot_to_coco_tracking(
            seq_dir=seq_dir,
            ann_dir=ann_dir,
            output_json=os.path.join(
                output_dir, "coco_annotations", "tracking_all.json"
            ),
            area_percentile_thresholds=area_percentile_thresholds,
        )

        all_videos = full_coco["videos"]

        if len(all_videos) < 2:
            print(" Only 1 sequence — using all for train.")
            train_videos = all_videos
            val_videos = []
        else:
            train_videos, val_videos = train_test_split(
                all_videos,
                test_size=val_split,
                random_state=42,
            )

        train_vid_ids = {v["id"] for v in train_videos}
        val_vid_ids   = {v["id"] for v in val_videos}

        train_imgs = [
            i for i in full_coco["images"]
            if i["video_id"] in train_vid_ids
        ]
        val_imgs = [
            i for i in full_coco["images"]
            if i["video_id"] in val_vid_ids
        ]

        train_img_ids = {i["id"] for i in train_imgs}
        val_img_ids   = {i["id"] for i in val_imgs}

        train_anns = [
            a for a in full_coco["annotations"]
            if a["image_id"] in train_img_ids
        ]
        val_anns = [
            a for a in full_coco["annotations"]
            if a["image_id"] in val_img_ids
        ]

        train_tracks = [
            t for t in full_coco["tracks"]
            if t["video_id"] in train_vid_ids
        ]
        val_tracks = [
            t for t in full_coco["tracks"]
            if t["video_id"] in val_vid_ids
        ]

        shared = {
            "categories": full_coco["categories"],
            "info":       full_coco["info"],
            "licenses":   full_coco["licenses"],
        }

        train_coco = {
            **shared,
            "videos":      train_videos,
            "images":      train_imgs,
            "annotations": train_anns,
            "tracks":      train_tracks,
        }
        val_coco = {
            **shared,
            "videos":      val_videos,
            "images":      val_imgs,
            "annotations": val_anns,
            "tracks":      val_tracks,
        }

        ann_out = os.path.join(output_dir, "coco_annotations")
        os.makedirs(ann_out, exist_ok=True)

        train_json = os.path.join(ann_out, "tracking_train.json")
        val_json   = os.path.join(ann_out, "tracking_val.json")

        with open(train_json, 'w') as f:
            json.dump(train_coco, f, indent=2)
        with open(val_json, 'w') as f:
            json.dump(val_coco, f, indent=2)

        print(f"\n{'='*60}")
        print("Train/Val Split (by video sequence)")
        print(f"{'='*60}")
        print(f"  Train: {len(train_videos)} videos, "
              f"{len(train_imgs)} frames, "
              f"{len(train_anns)} annotations, "
              f"{len(train_tracks)} tracks")
        print(f"  Val:   {len(val_videos)} videos, "
              f"{len(val_imgs)} frames, "
              f"{len(val_anns)} annotations, "
              f"{len(val_tracks)} tracks")
        print(f"\n  Train: {[v['name'] for v in train_videos]}")
        print(f"  Val:   {[v['name'] for v in val_videos]}")
        print(f"\n  Saved: {train_json}")
        print(f"  Saved: {val_json}")

        return train_coco, val_coco

    @staticmethod
    def visualize_area_distributions(
        json_path: str,
        save_path: str = None,
    ):
        with open(json_path, 'r') as f:
            coco = json.load(f)

        categories = coco["categories"]
        cat_id_to_name = {c["id"]: c["name"] for c in categories}

        areas_by_cat = defaultdict(list)
        for ann in coco["annotations"]:
            cid = ann["category_id"]
            area = ann.get("area", ann["bbox"][2] * ann["bbox"][3])
            areas_by_cat[cid].append(area)

        filter_info = coco.get("info", {}).get("bbox_filter", {})
        abs_thresholds = filter_info.get("absolute_area_thresholds", {})

        n_cats = len(categories)
        fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 4))
        if n_cats == 1:
            axes = [axes]

        for ax, cat in zip(axes, categories):
            cid = cat["id"]
            cname = cat["name"]
            areas = areas_by_cat.get(cid, [])

            if not areas:
                ax.set_title(f"{cname}\n(no data)")
                continue

            areas = np.array(areas)
            ax.hist(areas, bins=80, color='steelblue', alpha=0.7,
                    edgecolor='white', linewidth=0.3)

            thresh = abs_thresholds.get(cname, 0.0)
            if thresh > 0:
                ax.axvline(thresh, color='red', linestyle='--', linewidth=2,
                           label=f'Threshold: {thresh:.0f} px²')
                ax.legend(fontsize=8)

            ax.set_title(f"{cname}\n(n={len(areas):,}, "
                         f"median={np.median(areas):.0f} px²)", fontsize=10)
            ax.set_xlabel("Area (px²)")
            ax.set_ylabel("Count")
            ax.set_xlim(left=0)

        plt.suptitle("Per-Class Bounding Box Area Distributions",
                      fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved area distribution plot to: {save_path}")

        plt.show()

    @staticmethod
    def verify_tracking_coco(
        json_path: str,
        img_dir: str,
        num_frames: int = 8,
        video_name: str = None,
        start_frame: int = None,
        save_path: str = None,
    ):
        with open(json_path, 'r') as f:
            coco = json.load(f)

        cat_lookup = {c["id"]: c["name"] for c in coco["categories"]}

        videos = coco.get("videos", [])
        if not videos:
            raise ValueError(
                "No 'videos' key found. Is this a tracking JSON?"
            )

        if video_name:
            video = next(
                (v for v in videos if v["name"] == video_name), None
            )
            if not video:
                avail = [v['name'] for v in videos]
                raise ValueError(
                    f"Video '{video_name}' not found. Available: {avail}"
                )
        else:
            video = rnd.choice(videos)

        vid_id = video["id"]
        print(f"Visualizing: {video['name']} "
              f"({video.get('n_frames', '?')} frames)")

        vid_images = sorted(
            [i for i in coco["images"] if i["video_id"] == vid_id],
            key=lambda x: x["frame_id"]
        )

        if not vid_images:
            print(" No frames found for this video.")
            return

        max_start = max(0, len(vid_images) - num_frames)
        if start_frame is not None:
            start_idx = min(start_frame, max_start)
        else:
            start_idx = rnd.randint(0, max_start) if max_start > 0 else 0

        selected = vid_images[start_idx:start_idx + num_frames]

        anns_by_img = {}
        for ann in coco["annotations"]:
            iid = ann["image_id"]
            if iid not in anns_by_img:
                anns_by_img[iid] = []
            anns_by_img[iid].append(ann)

        np.random.seed(42)
        track_colors = {}

        def get_color(track_id):
            if track_id not in track_colors:
                c = tuple(np.random.randint(60, 255, 3).tolist())
                track_colors[track_id] = tuple(v / 255.0 for v in c)
            return track_colors[track_id]

        n_cols = min(4, len(selected))
        n_rows = (len(selected) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 6 * n_rows)
        )
        axes = np.array(axes).flatten()

        for idx, (ax, img_info) in enumerate(zip(axes, selected)):
            img_path = os.path.join(img_dir, img_info["file_name"])

            try:
                img = Image.open(img_path)
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f"Cannot load\n{e}",
                        transform=ax.transAxes, ha='center')
                ax.set_title(f"Frame {img_info['frame_id']}")
                ax.axis('off')
                continue

            anns = anns_by_img.get(img_info["id"], [])

            for ann in anns:
                x, y, w, h = ann["bbox"]
                track_id = ann.get("track_id", -1)
                cat_name = cat_lookup.get(ann["category_id"], "?")
                color = get_color(track_id)

                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)

                label = f"T{track_id} {cat_name}"
                ax.text(
                    x, y - 4, label,
                    color='white',
                    fontsize=7,
                    fontweight='bold',
                    bbox=dict(
                        facecolor=color,
                        alpha=0.8,
                        pad=1.5,
                        boxstyle='round,pad=0.2'
                    )
                )

            ax.set_title(
                f"Frame {img_info['frame_id']} "
                f"({len(anns)} objects)",
                fontsize=10
            )
            ax.axis('off')

        for ax in axes[len(selected):]:
            ax.axis('off')

        plt.suptitle(
            f"VisDrone Tracking: {video['name']}\n"
            f"Frames {selected[0]['frame_id']}–{selected[-1]['frame_id']} "
            f"| Same color = same track ID",
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()