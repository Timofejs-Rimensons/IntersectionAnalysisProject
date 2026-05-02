import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random as rnd
import os

import json

import cv2

from services.TrafficFlowEstimator import TrafficFlowEstimator

class Utils:
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    COLORS = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8',
        '#f58231', '#911eb4', '#42d4f4', '#f032e6',
        '#bfef45', '#fabed4', '#469990', '#dcbeff',
        '#9A6324', '#800000', '#aaffc3', '#808000',
    ]

    def __init__(self, config=None):
        self.config = config


    def denormalize(self, tensor):
        img = tensor.cpu().clone().permute(1, 2, 0).numpy()
        img = img * self.IMAGENET_STD + self.IMAGENET_MEAN
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def draw_boxes(self, ax, boxes, labels, cat_lookup=None, title=""):
        if cat_lookup is None:
            cat_lookup = {}
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            cat_id = int(label)
            cat_name = cat_lookup.get(cat_id, f"cls_{cat_id}")
            color = self.COLORS[cat_id % len(self.COLORS)]
            rect = patches.Rectangle(
                (x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 4, f"{cat_name}",
                color='white', fontsize=7, fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.8, pad=1.5, boxstyle='round,pad=0.2')
            )
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    def verify_dataset(self, dataset, num_samples=6, cat_lookup=None, save_path=None):
        n = min(num_samples, len(dataset))
        indices = rnd.sample(range(len(dataset)), n)
        n_cols = min(3, n)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
        axes = np.array(axes).flatten()

        for plot_idx, data_idx in enumerate(indices):
            ax = axes[plot_idx]
            image, target = dataset[data_idx]
            boxes = target.get("boxes", torch.zeros(0, 4)).numpy()
            labels = target.get("labels", torch.zeros(0, dtype=torch.int64)).numpy()
            img_np = self.denormalize(image)
            ax.imshow(img_np)
            title = f"idx={data_idx} | {tuple(image.shape)} | {len(boxes)} boxes"
            self.draw_boxes(ax, boxes, labels, cat_lookup, title)

        for ax in axes[len(indices):]:
            ax.axis('off')

        plt.suptitle(f"Augmented Dataset ({n} samples)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def verify_augmentation_variety(self, dataset, sample_idx=0, num_repeats=8, cat_lookup=None, save_path=None):
        n_cols = min(4, num_repeats)
        n_rows = (num_repeats + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
        axes = np.array(axes).flatten()

        for i in range(num_repeats):
            ax = axes[i]
            image, target = dataset[sample_idx]
            boxes = target.get("boxes", torch.zeros(0, 4)).numpy()
            labels = target.get("labels", torch.zeros(0, dtype=torch.int64)).numpy()
            img_np = self.denormalize(image)
            ax.imshow(img_np)
            self.draw_boxes(ax, boxes, labels, cat_lookup, title=f"Attempt {i + 1} | {len(boxes)} boxes")

        for ax in axes[num_repeats:]:
            ax.axis('off')

        plt.suptitle(f"Augmentation Variety (idx={sample_idx})", fontsize=13, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def verify_dataloader(self, dataloader, num_batches=1, max_per_batch=8, cat_lookup=None, save_path=None):
        data_iter = iter(dataloader)

        for batch_idx in range(num_batches):
            try:
                images, targets = next(data_iter)
            except StopIteration:
                break

            n_show = min(max_per_batch, len(images))
            n_cols = min(4, n_show)
            n_rows = (n_show + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
            axes = np.array(axes).flatten()

            for i in range(n_show):
                ax = axes[i]
                image = images[i]
                target = targets[i]
                boxes = target.get("boxes", torch.zeros(0, 4)).cpu().numpy()
                labels = target.get("labels", torch.zeros(0, dtype=torch.int64)).cpu().numpy()
                img_id = target.get("image_id", "?")
                img_np = self.denormalize(image)
                ax.imshow(img_np)
                title = f"img_id={img_id} | {tuple(image.shape)} | {len(boxes)} boxes"
                self.draw_boxes(ax, boxes, labels, cat_lookup, title)

            for ax in axes[n_show:]:
                ax.axis('off')

            plt.suptitle(f"DataLoader Batch {batch_idx} (size={len(images)})", fontsize=13, fontweight='bold')
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_batch{batch_idx}.png", dpi=150, bbox_inches='tight')
            plt.show()

    def verify_pipeline(self, dataset, dataloader, cat_lookup=None, save_dir=None):
        sp1, sp2, sp3 = None, None, None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            sp1 = os.path.join(save_dir, "dataset_samples.png")
            sp2 = os.path.join(save_dir, "augmentation_variety.png")
            sp3 = os.path.join(save_dir, "dataloader")

        self.verify_dataset(dataset, num_samples=6, cat_lookup=cat_lookup, save_path=sp1)
        self.verify_augmentation_variety(dataset, sample_idx=0, num_repeats=8, cat_lookup=cat_lookup, save_path=sp2)
        self.verify_dataloader(dataloader, num_batches=1, max_per_batch=8, cat_lookup=cat_lookup, save_path=sp3)

    def plot_training_history(self, history, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(history["train_loss"], label="Train Loss", marker='o')
        if "val_loss" in history and history["val_loss"]:
            epochs_val = list(range(0, len(history["val_loss"])))
            axes[0].plot(epochs_val, history["val_loss"], label="Val Loss", marker='s')
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history["lr"], marker='o', color='green')
        axes[1].set_title("Learning Rate")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True)

        if "mAP" in history and history["mAP"]:
            axes[2].plot(history["mAP"], marker='o', color='orange')
            axes[2].set_title("mAP@0.5:0.95")
            axes[2].set_xlabel("Eval Step")
            axes[2].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def visualize_predictions(self, model, dataset, config, num_samples=4, score_thresh=0.5):
        if self.config is None:
            if config is None:
                raise ValueError("Config object must be provided to Utils during initialization or directly to visualize_predictions method.")
            local_config = config
        else:
            local_config = self.config

        model.eval()
        device = torch.device(local_config.device)

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        fig, axes = plt.subplots(1, num_samples, figsize=(7 * num_samples, 7))
        if num_samples == 1:
            axes = [axes]

        indices = np.random.choice(len(dataset), num_samples, replace=False)

        for ax, idx in zip(axes, indices):
            image, target = dataset[idx]

            with torch.no_grad():
                prediction = model([image.to(device)])[0]

            img_display = (image.cpu() * self.IMAGENET_STD[:, None, None] + self.IMAGENET_MEAN[:, None, None]).clamp(0, 1)
            img_display = img_display.permute(1, 2, 0).numpy()

            ax.imshow(img_display)

            boxes = prediction["boxes"].cpu()
            scores = prediction["scores"].cpu()
            labels = prediction["labels"].cpu()

            for box, score, label in zip(boxes, scores, labels):
                if score < score_thresh:
                    continue
                x1, y1, x2, y2 = box.tolist()
                color = colors[label.item() % len(colors)]
                name = local_config.class_names[label.item()]

                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f"{name} {score:.2f}",
                        color='white', fontsize=8, fontweight='bold',
                        bbox=dict(facecolor=color, alpha=0.7, pad=2))

            ax.axis('off')

        plt.tight_layout()
        plt.show()
class TrafficSeriesVisualizer:

    COLORS = {
        'entry': '#2ecc71',
        'exit': '#3498db',
        'entry_alt': '#e74c3c',
        'exit_alt': '#f39c12',
        'historical': 'blue',
        'forecast': 'red',
        'smoothed': 'navy',
        'event': 'magenta',
    }
    PALETTE = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    def __init__(self, series_data=None, selected_series=None):
        self.series_data = series_data
        self.selected_series = selected_series

    def plot_time_series(self, road_id, road_data, ax=None,
                         mode='bars', bin_seconds=None, num_bins=None,
                         show_exits=True, smoothed=None, events=None, title=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 4))

        bin_seconds = bin_seconds if bin_seconds is not None else (
            self.series_data['bin_seconds'] if self.series_data else 1.0)
        num_bins = num_bins if num_bins is not None else len(road_data['time_series_entries'])

        entries = np.array(road_data['time_series_entries'])
        exits = np.array(road_data.get('time_series_exits', []))
        time_bins = np.arange(num_bins)
        times = time_bins * bin_seconds

        entry_dir = road_data.get('entry_direction', 'entry')
        exit_dir = road_data.get('exit_direction', 'exit')
        entry_color = self.COLORS['entry'] if entry_dir == 'from_left' else self.COLORS['entry_alt']
        exit_color = self.COLORS['exit'] if exit_dir == 'from_left' else self.COLORS['exit_alt']

        if mode == 'heatmap':
            traffic_matrix = np.vstack([entries, exits]) if len(exits) else entries[None, :]
            im = ax.imshow(traffic_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax.set_yticks(range(traffic_matrix.shape[0]))
            ax.set_yticklabels([entry_dir, exit_dir][:traffic_matrix.shape[0]], fontsize=9)
            ticks = np.linspace(0, num_bins - 1, min(11, num_bins))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t*bin_seconds:.1f}s" for t in ticks], rotation=45, fontsize=8)
            ax.set_xlabel('Time', fontsize=9, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('Vehicle Count', fontsize=8)

        elif mode == 'mirror':
            ax.bar(times, entries, width=bin_seconds * 0.9, alpha=0.45,
                   color='steelblue', label=f"Entries ({entry_dir})")
            if show_exits and len(exits):
                ax.bar(times, -exits, width=bin_seconds * 0.9, alpha=0.45,
                       color='salmon', label=f"Exits ({exit_dir})")
            if smoothed is not None:
                ax.plot(times, smoothed, color=self.COLORS['smoothed'], linewidth=2,
                        label='Entries (smoothed)')
            ax.axhline(0, color='black', linewidth=0.6)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Vehicles per bin')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        elif mode == 'line':
            ax.plot(times, entries, marker='o', color='steelblue', label='Entries')
            ax.bar(times, entries, width=bin_seconds * 0.85, alpha=0.35, color='steelblue')
            if events:
                fps = road_data.get('fps', 1)
                event_times = [(ev['frame'] - 1) / fps for ev in events]
                event_y = [max(entries) * 0.05 if max(entries) > 0 else 0.1 for _ in event_times]
                ax.scatter(event_times, event_y, c=self.COLORS['event'], s=40,
                           alpha=0.7, label='Entry events')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Vehicles per bin')
            ax.grid(True, alpha=0.3)
            ax.legend()

        else:
            bw = 0.35
            ax.bar(time_bins - bw/2, entries, bw, label=f'Entries ({entry_dir})',
                   color=entry_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            if show_exits and len(exits):
                ax.bar(time_bins + bw/2, exits, bw, label=f'Exits ({exit_dir})',
                       color=exit_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_ylabel('Count', fontsize=10, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_ylim(bottom=0)
            step = max(1, num_bins // 10)
            ax.set_xticks(time_bins[::step])
            ax.set_xticklabels([f"{t:.0f}s" for t in times[::step]], rotation=45)
            ax.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')

        if title is None:
            title = (f"Road {road_id} | Entries: {road_data.get('total_entries', '?')} "
                     f"| Exits: {road_data.get('total_exits', '?')}")
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        return ax

    def plot_all_roads(self, mode='bars', suptitle=None, figsize_per_road=(14, 3.5)):
        roads = self.series_data['roads']
        n = len(roads)
        fig, axes = plt.subplots(n, 1, figsize=(figsize_per_road[0], figsize_per_road[1] * n),
                                  squeeze=False)
        for idx, (road_id, road_data) in enumerate(roads.items()):
            self.plot_time_series(road_id, road_data, ax=axes[idx, 0], mode=mode,
                                  bin_seconds=self.series_data['bin_seconds'],
                                  num_bins=self.series_data['num_bins'])
        if suptitle is None:
            suptitle = f"Traffic Flow Time Series - {self.selected_series}"
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
        return fig, axes

    def plot_lines_with_series(self, video_path, denorm_lines=None, frame_idx=None,
                                series_mode='bars'):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if frame_idx is None:
            frame_idx = total_frames // 3
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        roads = self.series_data['roads']
        n = len(roads)
        fig = plt.figure(figsize=(16, 4 + 3 * n))
        gs = fig.add_gridspec(n + 1, 1, height_ratios=[3] + [1] * n)

        ax_frame = fig.add_subplot(gs[0, 0])
        ax_frame.imshow(frame_rgb)
        for road_id, road_data in roads.items():
            if denorm_lines is not None and int(road_id) < len(denorm_lines):
                (x1, y1), (x2, y2) = denorm_lines[int(road_id)]
            else:
                (x1, y1), (x2, y2) = road_data['line']
            ax_frame.plot([x1, x2], [y1, y2], color='cyan', linewidth=3, alpha=0.85)
            mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            ax_frame.text(mx, my, f"Road {road_id}", color='white', fontsize=11,
                          fontweight='bold', ha='center', va='center',
                          bbox=dict(facecolor='blue', alpha=0.6, pad=3))
        ax_frame.axis('off')
        ax_frame.set_title(
            f"{self.selected_series} - Frame {frame_idx}/{total_frames} ({frame_idx/fps:.1f}s)",
            fontsize=13, fontweight='bold')

        for idx, (road_id, road_data) in enumerate(roads.items()):
            ax = fig.add_subplot(gs[idx + 1, 0])
            self.plot_time_series(road_id, road_data, ax=ax, mode=series_mode,
                                  bin_seconds=self.series_data['bin_seconds'],
                                  num_bins=self.series_data['num_bins'])

        fig.suptitle(f"Roads & Time Series - {self.selected_series}",
                     fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_class_distribution(self):
        roads = self.series_data['roads']
        n = len(roads)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
        for idx, (road_id, road_data) in enumerate(roads.items()):
            ax = axes[0, idx]
            entries_by_class = road_data['entries_by_class']
            exits_by_class = road_data['exits_by_class']
            all_classes = sorted(set(list(entries_by_class.keys()) + list(exits_by_class.keys())))
            entry_counts = [entries_by_class.get(c, 0) for c in all_classes]
            exit_counts = [exits_by_class.get(c, 0) for c in all_classes]
            x_pos = np.arange(len(all_classes))
            bw = 0.35
            ax.bar(x_pos - bw/2, entry_counts, bw, label='Entry',
                   color=self.COLORS['entry'], alpha=0.8, edgecolor='black', linewidth=1)
            ax.bar(x_pos + bw/2, exit_counts, bw, label='Exit',
                   color=self.COLORS['exit'], alpha=0.8, edgecolor='black', linewidth=1)
            ax.set_xlabel('Vehicle Class', fontsize=10, fontweight='bold')
            ax.set_ylabel('Count', fontsize=10, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(all_classes, rotation=45, ha='right')
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_title(f"Road {road_id} Class Distribution", fontsize=11, fontweight='bold')
        fig.suptitle('Vehicle Class Distribution by Entry/Exit', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()


class TrafficSummaryReporter:

    def __init__(self, series_data=None, selected_series=None):
        self.series_data = series_data
        self.selected_series = selected_series

    def build_summary(self):
        summary_data = []
        for road_id, road_data in self.series_data['roads'].items():
            num_bins = self.series_data['num_bins']
            avg_in = road_data['total_entries'] / num_bins if num_bins else 0
            avg_out = road_data['total_exits'] / num_bins if num_bins else 0
            summary_data.append({
                'Road': road_id,
                'Entry Direction': road_data['entry_direction'],
                'Exit Direction': road_data['exit_direction'],
                'Total Entries': road_data['total_entries'],
                'Total Exits': road_data['total_exits'],
                'Avg Entry Rate': f"{avg_in:.2f}",
                'Avg Exit Rate': f"{avg_out:.2f}",
                'Entry Classes': ', '.join(f"{k}:{v}" for k, v in road_data['entries_by_class'].items()),
                'Exit Classes': ', '.join(f"{k}:{v}" for k, v in road_data['exits_by_class'].items()),
            })
        return summary_data

    def print_summary(self):
        summary_data = self.build_summary()
        print("\n" + "=" * 100)
        print(f"TRAFFIC FLOW SUMMARY - {self.selected_series}")
        print("=" * 100)
        for s in summary_data:
            print(f"\nRoad {s['Road']}")
            print(f"  Direction: {s['Entry Direction']} (entry) → {s['Exit Direction']} (exit)")
            print(f"  Entries: {s['Total Entries']} (avg {s['Avg Entry Rate']}/bin) | {s['Entry Classes']}")
            print(f"  Exits:   {s['Total Exits']} (avg {s['Avg Exit Rate']}/bin) | {s['Exit Classes']}")
        total = sum(s['Total Entries'] for s in summary_data)
        bs = self.series_data['bin_seconds']
        nb = self.series_data['num_bins']
        print(f"\n{'='*100}")
        print(f"Total vehicles tracked: {total}")
        print(f"Analysis period: {nb} bins × {bs:.3f}s = {nb*bs:.1f}s")
        print(f"{'='*100}")

    def print_flow_report(self, flow, target_seq):
        print("\n=== Flow frequency report (per direction) ===")
        print(f"Sequence: {target_seq}")
        print(f"  Duration: {flow['num_frames']/flow['fps']:.2f}s "
              f"({flow['num_frames']} frames @ {flow['fps']:.1f} fps)")
        print(f"  Bin size: {flow['bin_size_frames']} frames "
              f"({flow['bin_size_frames']/flow['fps']:.2f}s)")
        print(f"  Junction total entries: {flow['junction_total_entries']}")
        duration_min = flow['num_frames'] / flow['fps'] / 60.0
        for road_idx, road in flow['roads'].items():
            entries, exits = road['total_entries'], road['total_exits']
            rate_in = entries / duration_min if duration_min > 0 else 0.0
            rate_out = exits / duration_min if duration_min > 0 else 0.0
            peak = max(road['time_series_entries']) if road['time_series_entries'] else 0
            print(f"  Cross-section {road_idx}:")
            print(f"    Entry direction: {road['entry_direction']}  | Exit direction: {road['exit_direction']}")
            print(f"    Entries: {entries} ({rate_in:.2f}/min) | Exits: {exits} ({rate_out:.2f}/min)")
            print(f"    Net inflow: {entries - exits}")
            print(f"    Peak entries per bin: {peak}")
            print(f"    Unique entry tracks: {road['unique_entry_tracks']}")
            print(f"    By class: {road['entries_by_class']}")


class FlowEstimationRunner:

    def __init__(self, tracking_model, video_fps, vehicle_classes,
                 detection_conf_threshold, tracking_iou_threshold,
                 entry_direction='from_left'):
        self.estimator = TrafficFlowEstimator(
            tracking_model=tracking_model,
            entry_direction=entry_direction,
            fps=video_fps,
            bin_size_frames=video_fps,
            class_filter=vehicle_classes,
        )
        self.conf = detection_conf_threshold
        self.iou = tracking_iou_threshold

    def estimate(self, video_path, lines, normalized):
        return self.estimator.estimate_from_video(
            video_path=video_path, lines=lines, normalized=normalized,
            conf=self.conf, iou=self.iou,
            tracker='bytetrack.yaml', count_unique=False,
        )

    def summarize(self, flow):
        return self.estimator.summarize(flow)

    def smoothed(self, flow, window_bins=5):
        return self.estimator.smoothed_series(flow, window_bins=window_bins)

    def rate_per_minute(self, flow):
        return self.estimator.rate_per_minute(flow)


class LineDefinitionLoader:

    @staticmethod
    def load(json_path, sequence_name=None):
        with open(json_path, 'r') as f:
            data = json.load(f)
        sequences = data.get('sequences', {})
        if not sequences:
            raise ValueError(f"No sequences found in {json_path}")
        if sequence_name is None:
            sequence_name = next(iter(sequences))
        seq_info = sequences.get(sequence_name)
        if seq_info is None:
            raise ValueError(f"Sequence '{sequence_name}' not found")
        return (sequence_name, seq_info.get('lines', []),
                bool(seq_info.get('normalized', False)),
                seq_info.get('source'))

    @staticmethod
    def denormalize(lines, width, height, normalized=True):
        if normalized:
            return [[(int(x1*width), int(y1*height)),
                     (int(x2*width), int(y2*height))]
                    for [[x1, y1], [x2, y2]] in lines]
        return [[(int(x1), int(y1)), (int(x2), int(y2))]
                for [[x1, y1], [x2, y2]] in lines]

    @staticmethod
    def get_video_size(video_path):
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return w, h