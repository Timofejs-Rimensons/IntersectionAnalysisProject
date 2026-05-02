import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import numpy as np
import json
import yaml
import os
import gc
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
import random


class TrackingModel:
    def __init__(self, config, checkpoint_path=None, task='detect', verbose=False):
        self.config = config
        self.device = torch.device(config.device)
        self.classes = config.class_names
        self.colors = config.colors
        self.task = task
        self.verbose = verbose
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "box_loss": [],
            "cls_loss": [],
            "lr": [],
            "mAP50": [],
            "mAP": []
        }
        self.best_map = 0.0
        self.start_epoch = 0
        self.class_weights = None
        self.model = self._build_model(model=config.model, checkpoint_path=checkpoint_path)

    def _build_model(self, model="yolo26m", checkpoint_path=None):
        if checkpoint_path and os.path.exists(checkpoint_path):
            if self.verbose:
                print(f"Loading checkpoint from {checkpoint_path}")
            model = YOLO(checkpoint_path)
        else:
            if self.verbose:
                print(f"Loading pre-trained {model} model.")
            model = YOLO(f'models/{model}.pt')

        frozen_count = 0
        for name, param in model.model.named_parameters():
            if not param.requires_grad:
                param.requires_grad = True
                frozen_count += 1

        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(
            p.numel() for p in model.model.parameters() if p.requires_grad
        )

        if self.verbose:
            print(f"Parameters: {total_params:,} total | {trainable_params:,} trainable")

        if trainable_params == 0:
            raise RuntimeError(
                "Model has 0 trainable parameters after unfreezing."
            )

        return model

    @staticmethod
    def _tensor_to_int(t):
        if isinstance(t, int):
            return t
        if isinstance(t, np.ndarray):
            return int(t.item())
        if hasattr(t, 'dim'):
            return int(t.item() if t.dim() == 0 else t[0].item())
        return int(t)

    @staticmethod
    def _tensor_to_float(t):
        if isinstance(t, float):
            return t
        if isinstance(t, np.ndarray):
            return float(t.item())
        if hasattr(t, 'dim'):
            return float(t.item() if t.dim() == 0 else t[0].item())
        return float(t)

    @staticmethod
    def _compute_video_scale(width, height, max_dim=None):
        if max_dim is None:
            return 1.0
        max_side = max(width, height)
        return float(max_dim) / float(max_side) if max_side > max_dim else 1.0

    def _box_xyxy(self, box):
        coords = box.xyxy[0]
        if hasattr(coords, 'cpu'):
            coords = coords.cpu().numpy()
        else:
            coords = np.asarray(coords)
        return coords.astype(float).tolist()

    def _frame_centroid(self, box):
        coords = self._box_xyxy(box)
        x1, y1, x2, y2 = coords
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def compute_class_weights(self, labels_dir, method='inverse_freq', smoothing=1.0,
                              sample_ratio=0.2, use_cache=True):
        labels_dir = Path(labels_dir)

        cache_file = labels_dir.parent / f'.class_weights_{method}.json'
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                    if cache.get('method') == method and cache.get('num_classes') == len(self.classes):
                        self.class_weights = np.array(cache['weights'], dtype=np.float64)
                        return self.class_weights
            except Exception:
                pass

        label_files = list(labels_dir.glob('*.txt'))
        if not label_files:
            raise FileNotFoundError(f"No .txt label files found in {labels_dir}")

        sample_size = max(1, int(len(label_files) * sample_ratio))
        if sample_size < len(label_files):
            label_files = random.sample(label_files, sample_size)

        n_classes = len(self.classes)
        counts = np.zeros(n_classes, dtype=np.float64)

        for lf in tqdm(label_files, desc="Scanning labels", disable=len(label_files) < 100):
            try:
                with open(lf, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if not parts:
                            continue
                        try:
                            cls_id = int(parts[0])
                            if 0 <= cls_id < n_classes:
                                counts[cls_id] += 1
                        except (ValueError, IndexError):
                            continue
            except Exception:
                continue

        counts += smoothing
        total = counts.sum()

        if method == 'inverse_freq':
            weights = total / (n_classes * counts)
        elif method == 'sqrt_inverse':
            weights = np.sqrt(total / (n_classes * counts))
        elif method == 'effective_num':
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, counts)
            weights = (1.0 - beta) / effective_num
        elif method == 'uniform':
            weights = np.ones(n_classes, dtype=np.float64)
        else:
            raise ValueError(
                f"Unknown method '{method}'."
            )

        weights = weights / weights.mean()
        self.class_weights = weights

        try:
            cache_data = {
                'method': method,
                'num_classes': n_classes,
                'weights': weights.tolist(),
                'classes': self.classes
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass

        return weights

    def _apply_class_weights_to_model(self):
        if self.class_weights is None:
            return

        w = torch.tensor(self.class_weights, dtype=torch.float32).to(self.device)
        try:
            criterion = self.model.model.criterion
            if hasattr(criterion, 'bce_cls'):
                criterion.bce_cls = torch.nn.BCEWithLogitsLoss(
                    pos_weight=w, reduction='none')
            elif hasattr(criterion, 'cls_pw'):
                criterion.cls_pw = w
        except AttributeError:
            pass

    def train(self, train_data_yaml, val_data_yaml=None, resume_from_last=True,
              run_dir=None, class_weights_labels_dir=None,
              class_weight_method='sqrt_inverse', class_weight_sample_ratio=0.2):
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)

        if not os.path.exists(train_data_yaml):
            raise FileNotFoundError(f"Data YAML not found: {train_data_yaml}")

        with open(train_data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)

        if class_weights_labels_dir is not None:
            self.compute_class_weights(
                labels_dir=class_weights_labels_dir,
                method=class_weight_method,
                sample_ratio=class_weight_sample_ratio,
                use_cache=True,
            )
            if run_dir:
                wt_path = os.path.join(run_dir, 'class_weights.json')
                with open(wt_path, 'w') as f:
                    json.dump({
                        'method': class_weight_method,
                        'classes': self.classes,
                        'weights': self.class_weights.tolist(),
                    }, f, indent=2)

            _self = self

            def _on_train_start(trainer):
                _self._apply_class_weights_to_model()

            self.model.add_callback('on_train_start', _on_train_start)

        try:
            results = self.model.train(
                data=train_data_yaml,
                epochs=self.config.num_epochs,
                imgsz=self.config.img_size[0],
                batch=self.config.batch_size,
                patience=20,
                device=0 if self.device.type == 'cuda' else 'cpu',
                lr0=self.config.lr,
                lrf=0.01,
                momentum=0.937,
                weight_decay=self.config.weight_decay,
                warmup_epochs=5,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=10.0,
                cls=1.0,
                dfl=1.5,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=10.0,
                translate=0.2,
                freeze=10,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.0,
                copy_paste=0.0,
                auto_augment='randaugment',
                erasing=0.3,
                crop_fraction=1.0,
                val=True,
                split=0.1 if val_data_yaml is None else None,
                save=True,
                save_period=5,
                save_dir=run_dir if run_dir else self.config.output_dir,
                project='',
                name='',
                exist_ok=True,
                pretrained=True,
                optimizer='AdamW',
                seed=0,
                deterministic=True,
                single_cls=False,
                rect=False,
                cos_lr=True,
                close_mosaic=10,
                resume=resume_from_last,
                verbose=False,
            )

            self.last_run_dir = run_dir
            return results

        except KeyboardInterrupt:
            raise
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def detect(self, source, conf=0.5, iou=0.45):
        return self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=0 if self.device.type == 'cuda' else 'cpu',
            imgsz=self.config.img_size[0],
            verbose=self.verbose,
        )

    def track(self, source, conf=0.5, iou=0.45, tracker='bytetrack.yaml'):
        return self.model.track(
            source=source,
            conf=conf,
            iou=iou,
            device=0 if self.device.type == 'cuda' else 'cpu',
            imgsz=self.config.img_size[0],
            tracker=tracker,
            persist=True,
            verbose=self.verbose,
        )

    def track_stream(self, video_path, conf=0.5, iou=0.45, tracker='bytetrack.yaml',
                     max_buffer_frames=1, resize_scale=None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_results = []
        frame_buffer = []
        frame_indices = []

        try:
            with tqdm(total=frame_count, desc=f"Tracking {os.path.basename(video_path)}",
                     unit='frame', disable=not self.verbose) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        if frame_buffer:
                            buffer_results = self._process_frame_buffer(
                                frame_buffer, frame_indices, conf, iou, tracker, resize_scale
                            )
                            all_results.extend(buffer_results)
                        break

                    if resize_scale and resize_scale != 1.0:
                        h, w = frame.shape[:2]
                        new_h, new_w = int(h * resize_scale), int(w * resize_scale)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                    frame_buffer.append(frame)
                    frame_indices.append(len(all_results) + len(frame_buffer) - 1)

                    if len(frame_buffer) >= max_buffer_frames:
                        buffer_results = self._process_frame_buffer(
                            frame_buffer, frame_indices, conf, iou, tracker, resize_scale
                        )
                        all_results.extend(buffer_results)
                        frame_buffer = []
                        frame_indices = []
                        pbar.update(max_buffer_frames)

                        gc.collect()
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
        finally:
            cap.release()

        return all_results

    def _process_frame_buffer(self, frame_buffer, frame_indices, conf, iou, tracker, resize_scale):
        results = []
        for frame in frame_buffer:
            try:
                frame_result = self.model.track(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    device=0 if self.device.type == 'cuda' else 'cpu',
                    imgsz=self.config.img_size[0],
                    tracker=tracker,
                    persist=True,
                    verbose=False
                )
                results.extend(frame_result)
            except Exception:
                results.append(None)
        return results

    def _track_video_robust(self, video_path, conf=0.5, iou=0.45,
                            tracker='bytetrack.yaml', max_dim=1280,
                            skip_bad_frames=True):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        scale = self._compute_video_scale(orig_w, orig_h, max_dim)

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        results = []
        frame_idx = 0
        bad_frames = 0
        consecutive_failures = 0
        max_consecutive_failures = 30

        try:
            with tqdm(total=n_frames if n_frames > 0 else None,
                      desc=f"Tracking {os.path.basename(video_path)}",
                      unit='frame', disable=not self.verbose) as pbar:
                while True:
                    try:
                        ret, frame = cap.read()
                    except cv2.error:
                        ret, frame = False, None

                    if not ret or frame is None:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            break
                        if n_frames > 0 and frame_idx >= n_frames:
                            break
                        if not skip_bad_frames:
                            break
                        bad_frames += 1
                        results.append(None)
                        frame_idx += 1
                        pbar.update(1)
                        continue

                    consecutive_failures = 0

                    if frame.ndim != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
                        bad_frames += 1
                        results.append(None)
                        frame_idx += 1
                        pbar.update(1)
                        continue

                    if scale != 1.0:
                        try:
                            frame = cv2.resize(frame, (new_w, new_h),
                                               interpolation=cv2.INTER_LINEAR)
                        except cv2.error:
                            bad_frames += 1
                            results.append(None)
                            frame_idx += 1
                            pbar.update(1)
                            continue

                    try:
                        frame_result = self.model.track(
                            source=frame,
                            conf=conf,
                            iou=iou,
                            device=0 if self.device.type == 'cuda' else 'cpu',
                            imgsz=self.config.img_size[0],
                            tracker=tracker,
                            persist=True,
                            verbose=False,
                        )
                        if isinstance(frame_result, list) and frame_result:
                            results.append(frame_result[0])
                        else:
                            results.append(None)
                    except Exception:
                        bad_frames += 1
                        results.append(None)

                    frame_idx += 1
                    pbar.update(1)

                    if frame_idx % 200 == 0:
                        gc.collect()
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
        finally:
            cap.release()
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        meta = {
            'orig_size': (orig_w, orig_h),
            'processed_size': (new_w, new_h),
            'scale': scale,
            'fps': video_fps,
            'declared_frames': n_frames,
            'processed_frames': frame_idx,
            'bad_frames': bad_frames,
        }
        return results, meta

    def extract_detections(self, results):
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'class_id': int(box.cls[0]),
                    'class_name': self.classes[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].cpu().numpy().tolist()
                })
        return detections

    def extract_tracks(self, results):
        tracks = []
        for result in results:
            if result.boxes.id is None:
                continue
            for box, track_id in zip(result.boxes, result.boxes.id):
                tracks.append({
                    'track_id': self._tensor_to_int(track_id),
                    'class_id': int(box.cls[0]),
                    'class_name': self.classes[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].cpu().numpy().tolist()
                })
        return tracks

    def visualize_detections(self, image_path, results, save_path=None):
        img = cv2.imread(image_path)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                color = tuple(int(self.colors[class_id].lstrip('#')[i:i+2], 16)
                             for i in (0, 2, 4))[::-1]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{self.classes[class_id]} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if save_path:
            cv2.imwrite(save_path, img)
        return img

    def visualize_tracks(self, video_path, results, save_path=None, fps=30):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        track_colors = {}
        for result in results:
            _, frame = cap.read()
            if frame is None:
                break
            if result.boxes.id is None:
                continue
            for box, track_id in zip(result.boxes, result.boxes.id):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                tid = self._tensor_to_int(track_id)
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if tid not in track_colors:
                    track_colors[tid] = tuple(np.random.randint(0, 256, 3).tolist())
                color = track_colors[tid]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{tid} {self.classes[class_id]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if save_path:
                out.write(frame)
        cap.release()
        if save_path:
            out.release()

    def extract_track_history(self, results, class_filter=None):
        filter_ids = None
        filter_names = None
        if class_filter is not None:
            if all(isinstance(x, int) for x in class_filter):
                filter_ids = set(class_filter)
            else:
                filter_names = set(class_filter)

        track_history = {}
        for frame_idx, result in enumerate(results, start=1):
            if result is None or result.boxes.id is None:
                continue

            for box, track_id in zip(result.boxes, result.boxes.id):
                tid = self._tensor_to_int(track_id)
                class_id = int(box.cls[0])
                class_name = self.classes[class_id]
                if filter_ids is not None and class_id not in filter_ids:
                    continue
                if filter_names is not None and class_name not in filter_names:
                    continue

                box_xyxy = self._box_xyxy(box)
                centroid = ((box_xyxy[0] + box_xyxy[2]) / 2.0,
                            (box_xyxy[1] + box_xyxy[3]) / 2.0)

                track_history.setdefault(tid, []).append({
                    'frame': frame_idx,
                    'centroid': centroid,
                    'box': box_xyxy,
                    'class_id': class_id,
                    'class_name': class_name,
                })

        return track_history

    def _interpolate_track_history(self, track_history, max_gap_frames=3):
        if max_gap_frames is None or max_gap_frames <= 0:
            return track_history

        filled_history = {}
        for tid, history in track_history.items():
            sorted_history = sorted(history, key=lambda x: x['frame'])
            if not sorted_history:
                continue

            interpolated = [sorted_history[0]]
            for prev, curr in zip(sorted_history, sorted_history[1:]):
                gap = curr['frame'] - prev['frame']
                if 1 < gap <= max_gap_frames + 1:
                    for offset in range(1, gap):
                        alpha = float(offset) / float(gap)
                        prev_box = np.asarray(prev['box'], dtype=float)
                        curr_box = np.asarray(curr['box'], dtype=float)
                        interp_box = (prev_box + alpha * (curr_box - prev_box)).tolist()
                        interpolated.append({
                            'frame': prev['frame'] + offset,
                            'centroid': ((interp_box[0] + interp_box[2]) / 2.0,
                                         (interp_box[1] + interp_box[3]) / 2.0),
                            'box': interp_box,
                            'class_id': prev['class_id'],
                            'class_name': prev['class_name'],
                        })
                interpolated.append(curr)
            filled_history[tid] = interpolated

        return filled_history

    def _count_from_track_history(self, track_history, lines, count_unique=True, fps=30.0):
        if not lines or not isinstance(lines, (list, tuple)):
            raise ValueError("Lines must be a list of ((x1, y1), (x2, y2)) tuples")

        line_defs = [((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))
                     for p1, p2 in lines]

        def _empty_bucket():
            return {'total': 0, 'by_class': Counter(), 'track_ids': set(), 'events': []}

        line_summary = {
            idx: {
                'line': line,
                'total': 0,
                'by_class': Counter(),
                'track_ids': set(),
                'events': [],
                'from_left': _empty_bucket(),
                'from_right': _empty_bucket(),
            }
            for idx, line in enumerate(line_defs)
        }

        for tid, history in track_history.items():
            if len(history) < 2:
                continue
            cls_name = history[0]['class_name']
            cls_id = history[0]['class_id']

            for line_index, line in enumerate(line_defs):
                left_events = []
                right_events = []
                for prev, curr in zip(history, history[1:]):
                    side = self._segment_line_crossing_side(
                        prev['centroid'], curr['centroid'], line[0], line[1]
                    )
                    if side is None:
                        continue
                    event = {
                        'track_id': tid,
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'frame': curr['frame'],
                        'time_seconds': curr['frame'] / float(fps) if fps else None,
                        'prev_centroid': prev['centroid'],
                        'curr_centroid': curr['centroid'],
                        'direction': 'from_left' if side == 0 else 'from_right',
                    }
                    if side == 0:
                        left_events.append(event)
                    else:
                        right_events.append(event)

                if not left_events and not right_events:
                    continue

                entry = line_summary[line_index]

                def _add(bucket, events):
                    if count_unique:
                        if tid in bucket['track_ids']:
                            return
                        bucket['track_ids'].add(tid)
                        bucket['total'] += 1
                        bucket['by_class'][cls_name] += 1
                        bucket['events'].append(events[0])
                    else:
                        bucket['track_ids'].add(tid)
                        bucket['total'] += len(events)
                        bucket['by_class'][cls_name] += len(events)
                        bucket['events'].extend(events)

                if left_events:
                    _add(entry['from_left'], left_events)
                if right_events:
                    _add(entry['from_right'], right_events)

                if count_unique:
                    if tid not in entry['track_ids']:
                        entry['track_ids'].add(tid)
                        entry['total'] += 1
                        entry['by_class'][cls_name] += 1
                        entry['events'].append(
                            (left_events[0] if left_events else right_events[0])
                        )
                else:
                    entry['track_ids'].add(tid)
                    total_events = sorted(left_events + right_events,
                                          key=lambda e: e['frame'])
                    entry['total'] += len(total_events)
                    entry['by_class'][cls_name] += len(total_events)
                    entry['events'].extend(total_events)

        for entry in line_summary.values():
            entry['track_ids'] = sorted(entry['track_ids'])
            entry['by_class'] = dict(entry['by_class'])
            entry['events'].sort(key=lambda e: e['frame'])
            for key in ('from_left', 'from_right'):
                bucket = entry[key]
                bucket['track_ids'] = sorted(bucket['track_ids'])
                bucket['by_class'] = dict(bucket['by_class'])
                bucket['events'].sort(key=lambda e: e['frame'])

        return line_summary

    def count_objects_crossing_lines(self, results, lines, class_filter=None,
                                     count_unique=True, fps=30.0):
        track_history = self.extract_track_history(results, class_filter)
        interpolated_history = self._interpolate_track_history(
            track_history,
            max_gap_frames=getattr(self.config, 'track_gap_interpolation_frames', 0)
        )
        return self._count_from_track_history(interpolated_history, lines,
                                             count_unique=count_unique,
                                             fps=fps)

    def _orientation(self, a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1]) -
                (b[1] - a[1]) * (c[0] - a[0]))

    def _on_segment(self, a, b, c):
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

    def _segments_intersect(self, p1, p2, p3, p4):
        o1 = self._orientation(p1, p2, p3)
        o2 = self._orientation(p1, p2, p4)
        o3 = self._orientation(p3, p4, p1)
        o4 = self._orientation(p3, p4, p2)

        def _zero(val):
            return abs(val) < 1e-9

        if _zero(o1) and self._on_segment(p1, p2, p3):
            return True
        if _zero(o2) and self._on_segment(p1, p2, p4):
            return True
        if _zero(o3) and self._on_segment(p3, p4, p1):
            return True
        if _zero(o4) and self._on_segment(p3, p4, p2):
            return True

        return (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (
               o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0)

    def _segment_line_crossing_side(self, prev_pt, curr_pt, line_p1, line_p2):
        if not self._segments_intersect(prev_pt, curr_pt, line_p1, line_p2):
            return None

        ax, ay = line_p1
        bx, by = line_p2
        dx, dy = bx - ax, by - ay

        prev_side = dx * (prev_pt[1] - ay) - dy * (prev_pt[0] - ax)
        curr_side = dx * (curr_pt[1] - ay) - dy * (curr_pt[0] - ax)

        if prev_side > 0 and curr_side < 0:
            return 0
        if prev_side < 0 and curr_side > 0:
            return 1
        return None

    def annotate_video_with_lines(self, video_path, lines, results=None, output_path=None,
                                  show_tracks=False, line_colors=None, fps=None):
        if results is None:
            raise ValueError("Results are required to annotate video frames with lines")
        if not lines:
            raise ValueError("Please provide at least one line to draw")

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

        if line_colors is None:
            line_colors = [(0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 255, 0)]

        line_counts = self.count_objects_crossing_lines(results, lines, fps=video_fps)

        for frame_idx, result in enumerate(results):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            for line_index, line in enumerate(lines):
                color = line_colors[line_index % len(line_colors)]
                p1 = (int(line[0][0]), int(line[0][1]))
                p2 = (int(line[1][0]), int(line[1][1]))
                cv2.line(frame, p1, p2, color, 2)

                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                vx, vy = (p2[0] - p1[0]), (p2[1] - p1[1])
                norm = (vx ** 2 + vy ** 2) ** 0.5 or 1.0
                ux, uy = vx / norm, vy / norm
                arrow_end = (int(mid[0] + ux * 20), int(mid[1] + uy * 20))
                cv2.arrowedLine(frame, mid, arrow_end, color, 2, tipLength=0.5)

                summary = line_counts[line_index]
                left_n = summary['from_left']['total']
                right_n = summary['from_right']['total']
                label = (f"Line {line_index + 1}  "
                         f"L:{left_n}  R:{right_n}  T:{summary['total']}")
                cv2.putText(frame, label, (p1[0], max(20, p1[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            if show_tracks and result is not None and result.boxes.id is not None:
                for box, track_id in zip(result.boxes, result.boxes.id):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    tid = self._tensor_to_int(track_id)
                    color = line_colors[tid % len(line_colors)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{tid}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            if writer:
                writer.write(frame)

        cap.release()
        if writer:
            writer.release()
        return output_path

    def process_video_with_lines(self, source, lines, output_video=None, class_filter=None,
                                 conf=0.5, iou=0.45, tracker='bytetrack.yaml',
                                 count_unique=True, show_tracks=False, fps=30.0):
        if not lines or not isinstance(lines, (list, tuple)):
            raise ValueError("Lines must be a list of ((x1, y1), (x2, y2)) tuples")

        results = self.track(source, conf=conf, iou=iou, tracker=tracker)
        counts = self.count_objects_crossing_lines(
            results, lines,
            class_filter=class_filter,
            count_unique=count_unique,
            fps=fps,
        )
        if output_video:
            self.annotate_video_with_lines(
                video_path=source,
                lines=lines,
                results=results,
                output_path=output_video,
                show_tracks=show_tracks,
                fps=fps,
            )
        return {
            'video_path': source,
            'output_video': output_video,
            'line_counts': counts,
            'lines': lines,
            'class_filter': class_filter,
            'tracker': tracker,
            'frames_processed': len(results)
        }

    def analyze_crossings_from_video(self, video_path, lines, normalized=False,
                                     class_filter=None, conf=0.5, iou=0.45,
                                     tracker='bytetrack.yaml', count_unique=True,
                                     fps=None, max_dim=None, skip_bad_frames=True):
        if max_dim is None:
            max_dim = getattr(self.config, 'max_video_dim', 1280)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        probed_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        results, meta = self._track_video_robust(
            video_path=video_path,
            conf=conf,
            iou=iou,
            tracker=tracker,
            max_dim=max_dim,
            skip_bad_frames=skip_bad_frames,
        )

        proc_w, proc_h = meta['processed_size']
        scale = meta['scale']
        video_fps = fps or meta.get('fps') or probed_fps or 30.0

        if normalized:
            denorm_lines_orig = [
                ((int(p1[0] * orig_w), int(p1[1] * orig_h)),
                 (int(p2[0] * orig_w), int(p2[1] * orig_h)))
                for p1, p2 in lines
            ]
        else:
            denorm_lines_orig = [
                ((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))
                for p1, p2 in lines
            ]

        if scale != 1.0:
            scaled_lines = [
                ((int(p1[0] * scale), int(p1[1] * scale)),
                 (int(p2[0] * scale), int(p2[1] * scale)))
                for p1, p2 in denorm_lines_orig
            ]
        else:
            scaled_lines = denorm_lines_orig

        line_counts = self.count_objects_crossing_lines(
            results, scaled_lines,
            class_filter=class_filter,
            count_unique=count_unique,
            fps=video_fps,
        )

        overall_total = sum(s['total'] for s in line_counts.values())
        overall_left = sum(s['from_left']['total'] for s in line_counts.values())
        overall_right = sum(s['from_right']['total'] for s in line_counts.values())

        per_line = {
            idx: {
                'total': s['total'],
                'from_left': s['from_left']['total'],
                'from_right': s['from_right']['total'],
            }
            for idx, s in line_counts.items()
        }

        by_class_total = Counter()
        by_class_left = Counter()
        by_class_right = Counter()
        for s in line_counts.values():
            by_class_total.update(s['by_class'])
            by_class_left.update(s['from_left']['by_class'])
            by_class_right.update(s['from_right']['by_class'])

        all_events_left = []
        all_events_right = []
        for idx, s in line_counts.items():
            for ev in s['from_left']['events']:
                all_events_left.append({**ev, 'line_index': idx})
            for ev in s['from_right']['events']:
                all_events_right.append({**ev, 'line_index': idx})
        all_events_left.sort(key=lambda e: e['frame'])
        all_events_right.sort(key=lambda e: e['frame'])

        return {
            'video_path': video_path,
            'num_frames': len(results) if results else n_frames_meta,
            'frame_size': (orig_w, orig_h),
            'processed_frame_size': (proc_w, proc_h),
            'scale': scale,
            'fps': video_fps,
            'lines': denorm_lines_orig,
            'lines_processed': scaled_lines,
            'normalized_input': bool(normalized),
            'class_filter': class_filter,
            'results': results,
            'line_counts': line_counts,
            'crossings': {
                'from_left': all_events_left,
                'from_right': all_events_right,
            },
            'totals': {
                'overall': overall_total,
                'from_left': overall_left,
                'from_right': overall_right,
                'by_class': dict(by_class_total),
                'by_class_left': dict(by_class_left),
                'by_class_right': dict(by_class_right),
                'per_line': per_line,
            },
            'tracker': tracker,
            'robust_meta': meta,
        }

    def analyze_crossings_from_folder(self, frames_folder, boundaries,
                                      sequence_name=None, class_filter=None,
                                      conf=0.5, iou=0.45, tracker='bytetrack.yaml',
                                      count_unique=True, fps=30.0):
        if not os.path.isdir(frames_folder):
            raise FileNotFoundError(f"Frames folder not found: {frames_folder}")
        if not isinstance(boundaries, dict) and not isinstance(boundaries, list):
            raise ValueError("boundaries must be a dict or list")

        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        frame_files = sorted(
            f for f in os.listdir(frames_folder) if f.lower().endswith(valid_ext)
        )
        if not frame_files:
            raise RuntimeError(f"No image frames found in {frames_folder}")
        frame_paths = [os.path.join(frames_folder, f) for f in frame_files]

        first = cv2.imread(frame_paths[0])
        if first is None:
            raise RuntimeError(f"Cannot read first frame: {frame_paths[0]}")
        img_h, img_w = first.shape[:2]

        seq_key = sequence_name or os.path.basename(os.path.normpath(frames_folder))
        raw_lines, normalized, resolved_key = self._extract_lines_from_boundary(
            boundaries, seq_key
        )

        if normalized:
            lines = [
                ((int(p1[0] * img_w), int(p1[1] * img_h)),
                 (int(p2[0] * img_w), int(p2[1] * img_h)))
                for p1, p2 in raw_lines
            ]
        else:
            lines = [
                ((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))
                for p1, p2 in raw_lines
            ]

        results = self.track(source=frames_folder, conf=conf, iou=iou, tracker=tracker)

        line_counts = self.count_objects_crossing_lines(
            results, lines,
            class_filter=class_filter,
            count_unique=count_unique,
            fps=fps,
        )

        overall_total = sum(s['total'] for s in line_counts.values())
        overall_left = sum(s['from_left']['total'] for s in line_counts.values())
        overall_right = sum(s['from_right']['total'] for s in line_counts.values())

        per_line = {
            idx: {
                'total': s['total'],
                'from_left': s['from_left']['total'],
                'from_right': s['from_right']['total'],
            }
            for idx, s in line_counts.items()
        }

        by_class_total = Counter()
        by_class_left = Counter()
        by_class_right = Counter()
        for s in line_counts.values():
            by_class_total.update(s['by_class'])
            by_class_left.update(s['from_left']['by_class'])
            by_class_right.update(s['from_right']['by_class'])

        all_events_left = []
        all_events_right = []
        for idx, s in line_counts.items():
            for ev in s['from_left']['events']:
                all_events_left.append({**ev, 'line_index': idx})
            for ev in s['from_right']['events']:
                all_events_right.append({**ev, 'line_index': idx})
        all_events_left.sort(key=lambda e: e['frame'])
        all_events_right.sort(key=lambda e: e['frame'])

        return {
            'sequence_name': resolved_key,
            'frames_folder': frames_folder,
            'num_frames': len(frame_paths),
            'frame_size': (img_w, img_h),
            'fps': fps,
            'lines': lines,
            'normalized_input': bool(normalized),
            'class_filter': class_filter,
            'line_counts': line_counts,
            'crossings': {
                'from_left': all_events_left,
                'from_right': all_events_right,
            },
            'totals': {
                'overall': overall_total,
                'from_left': overall_left,
                'from_right': overall_right,
                'by_class': dict(by_class_total),
                'by_class_left': dict(by_class_left),
                'by_class_right': dict(by_class_right),
                'per_line': per_line,
            },
            'tracker': tracker,
        }

    @staticmethod
    def _extract_lines_from_boundary(boundary_data, seq_key):
        if isinstance(boundary_data, list):
            return boundary_data, False, seq_key

        if not isinstance(boundary_data, dict):
            raise ValueError(f"Unsupported boundary type: {type(boundary_data)}")

        if 'lines' in boundary_data and isinstance(boundary_data['lines'], list):
            return (boundary_data['lines'],
                    bool(boundary_data.get('normalized', False)),
                    seq_key)

        if seq_key in boundary_data:
            entry = boundary_data[seq_key]
            resolved = seq_key
        else:
            resolved = next(iter(boundary_data))
            entry = boundary_data[resolved]

        if isinstance(entry, list):
            return entry, False, resolved
        if isinstance(entry, dict) and 'lines' in entry:
            return (entry['lines'],
                    bool(entry.get('normalized', False)),
                    resolved)

        raise ValueError(
            f"Could not find 'lines' for sequence '{seq_key}'."
        )

    def select_lines_interactively(self, source=None, frame=None, num_lines=1,
                                   start_frame=0, title='Select lines'):
        if frame is None:
            if source is None:
                raise ValueError("Either source or frame must be provided")
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise ValueError(f"Unable to read frame {start_frame} from {source}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(frame_rgb)
        ax.set_title(f"{title} - click {num_lines*2} points")
        ax.axis('off')

        points = plt.ginput(num_lines * 2, timeout=0)
        plt.close(fig)

        if len(points) != num_lines * 2:
            raise RuntimeError(f"Expected {num_lines * 2} points, got {len(points)}")

        return [
            ((int(points[i][0]), int(points[i][1])),
             (int(points[i + 1][0]), int(points[i + 1][1])))
            for i in range(0, len(points), 2)
        ]

    def export_detections_to_coco(self, results, output_json, image_dir):
        coco = {
            "images": [], "annotations": [],
            "categories": [{"id": i, "name": name, "supercategory": "none"}
                           for i, name in enumerate(self.classes)],
            "info": {"description": "YOLO detection results", "version": "1.0", "year": 2024},
            "licenses": []
        }
        annotation_id = 1
        for image_id, result in enumerate(results, 1):
            if result is None:
                continue
            coco["images"].append({
                "id": image_id, "file_name": os.path.basename(result.path),
                "width": result.orig_shape[1], "height": result.orig_shape[0]
            })
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                coco["annotations"].append({
                    "id": annotation_id, "image_id": image_id,
                    "category_id": int(box.cls[0]), "bbox": bbox,
                    "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0,
                    "confidence": float(box.conf[0])
                })
                annotation_id += 1
        os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(coco, f, indent=2)
        return coco

    def export_tracks_to_mot(self, results, output_txt, video_name=None):
        os.makedirs(os.path.dirname(output_txt) or '.', exist_ok=True)
        with open(output_txt, 'w') as f:
            for frame_idx, result in enumerate(results, 1):
                if result is None or result.boxes.id is None:
                    continue
                for box, track_id in zip(result.boxes, result.boxes.id):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    tid = self._tensor_to_int(track_id)
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    f.write(f"{frame_idx},{tid},{x1:.0f},{y1:.0f},{w:.0f},{h:.0f},"                            f"{conf:.2f},{class_id},1\n")

    def export_tracks_to_coco_tracking(self, results, output_json, image_dir, video_name):
        coco_tracking = {
            "videos": [{"id": 1, "name": video_name,
                        "width": results[0].orig_shape[1] if results and results[0] is not None else 0,
                        "height": results[0].orig_shape[0] if results and results[0] is not None else 0,
                        "n_frames": len(results)}],
            "images": [], "annotations": [], "tracks": [],
            "categories": [{"id": i, "name": name, "supercategory": "none"}
                           for i, name in enumerate(self.classes)],
            "info": {"description": "YOLO tracking results", "version": "1.0", "year": 2024},
            "licenses": []
        }

        track_ids = set()
        for result in results:
            if result is None or result.boxes.id is None:
                continue
            for tid in result.boxes.id:
                track_ids.add(self._tensor_to_int(tid))
        for tid in sorted(track_ids):
            coco_tracking["tracks"].append({"id": tid, "video_id": 1, "category_id": 0})

        annotation_id = 1
        for frame_idx, result in enumerate(results, 1):
            if result is None:
                continue
            coco_tracking["images"].append({
                "id": frame_idx, "file_name": os.path.basename(result.path),
                "video_id": 1, "frame_id": frame_idx,
                "width": result.orig_shape[1], "height": result.orig_shape[0]
            })
            if result.boxes.id is None:
                continue
            for box, track_id in zip(result.boxes, result.boxes.id):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                tid = self._tensor_to_int(track_id)
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                area = float((x2 - x1) * (y2 - y1))
                coco_tracking["annotations"].append({
                    "id": annotation_id, "image_id": frame_idx,
                    "category_id": class_id, "track_id": tid,
                    "bbox": bbox, "area": area,
                    "iscrowd": 0, "confidence": conf
                })
                annotation_id += 1

        os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(coco_tracking, f, indent=2)
        return coco_tracking

    def save(self, checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
        self.model.save(checkpoint_path)

    def load(self, checkpoint_path):
        self.model = YOLO(checkpoint_path)

    def save_training_metrics(self, run_dir, results=None):
        os.makedirs(run_dir, exist_ok=True)
        metrics_files = ['results.png', 'results.csv',
                         'confusion_matrix.png', 'confusion_matrix_normalized.png']
        saved_files = [f for f in metrics_files
                       if os.path.exists(os.path.join(run_dir, f))]
        return saved_files

    def generate_example_predictions(self, run_dir, sample_images=None,
                                     num_samples=5, conf=0.5):
        os.makedirs(run_dir, exist_ok=True)
        examples_dir = os.path.join(run_dir, 'examples_predictions')
        os.makedirs(examples_dir, exist_ok=True)
        if not sample_images:
            return []
        sample_images = sample_images[:num_samples]
        saved_examples = []
        for idx, img_path in enumerate(sample_images, 1):
            if not os.path.exists(img_path):
                continue
            try:
                results = self.model.predict(
                    source=img_path, conf=conf,
                    device=0 if self.device.type == 'cuda' else 'cpu',
                    verbose=False
                )
                output_path = os.path.join(examples_dir, f'example_{idx:03d}.jpg')
                if results[0].save():
                    results[0].save(output_path)
                    saved_examples.append(output_path)
            except Exception:
                continue
        return saved_examples

    def create_run_summary(self, run_dir, data_yaml=None, model_name='yolo26m'):
        summary_path = os.path.join(run_dir, 'run_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("YOLO TRAINING RUN SUMMARY\n")
            f.write(f"Run Directory: {run_dir}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Task: {self.task}\n")
            f.write(f"Data YAML: {data_yaml}\n")
            f.write(f"Batch Size: {self.config.batch_size}\n")
            f.write(f"Epochs: {self.config.num_epochs}\n")
            f.write(f"Learning Rate: {self.config.lr}\n")
            f.write(f"Image Size: {self.config.img_size}\n")
            f.write(f"Device: {self.device}\n")
            if self.class_weights is not None:
                for name, w in zip(self.classes, self.class_weights):
                    f.write(f"{name}: {w:.4f}\n")
        return summary_path