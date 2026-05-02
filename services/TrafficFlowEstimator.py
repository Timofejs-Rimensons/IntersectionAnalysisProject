import json
import os
import numpy as np
from collections import defaultdict, Counter


class TrafficFlowEstimator:
    DEFAULT_CLASS = 'car'

    def __init__(self, tracking_model=None, entry_direction='from_left', fps=30.0,
                 bin_size_frames=1, class_filter=None):
        self.tracking_model = tracking_model
        self.entry_direction = entry_direction
        self.fps = float(fps)
        self.bin_size_frames = max(1, int(bin_size_frames))
        self.class_filter = class_filter

        if entry_direction not in ('from_left', 'from_right'):
            raise ValueError(
                f"entry_direction must be 'from_left' or 'from_right', got '{entry_direction}'"
            )

    def estimate_flow(self, results, lines, num_frames=None,
                      entry_direction=None, line_directions=None,
                      count_unique=False):
        if not lines:
            raise ValueError("At least one line is required")

        default_dir = entry_direction or self.entry_direction
        per_line_dir = {}
        for idx in range(len(lines)):
            per_line_dir[idx] = (line_directions or {}).get(idx, default_dir)

        line_counts = self.tracking_model.count_objects_crossing_lines(
            results, lines, class_filter=self.class_filter,
            count_unique=count_unique, fps=self.fps,
        )

        if num_frames is None:
            num_frames = self._infer_num_frames(results, line_counts)

        num_bins = int(np.ceil(num_frames / self.bin_size_frames)) if num_frames else 0
        roads = self._build_roads(line_counts, per_line_dir, num_bins)
        total_entries_series = self._sum_series(
            [r['time_series_entries'] for r in roads.values()])

        return {
            'fps': self.fps,
            'bin_size_frames': self.bin_size_frames,
            'num_frames': num_frames,
            'num_bins': num_bins,
            'bin_times_seconds': [(i * self.bin_size_frames) / self.fps for i in range(num_bins)],
            'default_entry_direction': default_dir,
            'per_line_entry_direction': per_line_dir,
            'class_filter': self.class_filter,
            'roads': roads,
            'junction_total_entries_series': total_entries_series,
            'junction_total_entries': sum(r['total_entries'] for r in roads.values()),
        }

    def estimate_from_folder(self, frames_folder, boundaries, sequence_name=None,
                             conf=0.5, iou=0.45, tracker='bytetrack.yaml',
                             entry_direction=None, line_directions=None,
                             count_unique=False):
        analysis = self.tracking_model.analyze_crossings_from_folder(
            frames_folder=frames_folder, boundaries=boundaries,
            sequence_name=sequence_name, class_filter=self.class_filter,
            conf=conf, iou=iou, tracker=tracker,
            count_unique=count_unique, fps=self.fps,
        )
        lines = analysis['lines']
        line_counts = analysis['line_counts']
        num_frames = analysis['num_frames']

        default_dir = entry_direction or self.entry_direction
        per_line_dir = {idx: (line_directions or {}).get(idx, default_dir)
                        for idx in range(len(lines))}

        num_bins = int(np.ceil(num_frames / self.bin_size_frames)) if num_frames else 0
        roads = self._build_roads(line_counts, per_line_dir, num_bins)
        total_entries_series = self._sum_series(
            [r['time_series_entries'] for r in roads.values()])

        return {
            'sequence_name': analysis['sequence_name'],
            'frames_folder': frames_folder,
            'fps': self.fps,
            'bin_size_frames': self.bin_size_frames,
            'num_frames': num_frames,
            'num_bins': num_bins,
            'frame_size': analysis['frame_size'],
            'bin_times_seconds': [(i * self.bin_size_frames) / self.fps for i in range(num_bins)],
            'default_entry_direction': default_dir,
            'per_line_entry_direction': per_line_dir,
            'class_filter': self.class_filter,
            'lines': lines,
            'roads': roads,
            'junction_total_entries_series': total_entries_series,
            'junction_total_entries': sum(r['total_entries'] for r in roads.values()),
        }

    def estimate_from_video(self, video_path, lines, normalized=False,
                            conf=0.5, iou=0.45, tracker='bytetrack.yaml',
                            entry_direction=None, line_directions=None,
                            count_unique=False, max_dim=1920,
                            skip_bad_frames=True):
        analysis = self.tracking_model.analyze_crossings_from_video(
            video_path=video_path, lines=lines, normalized=normalized,
            class_filter=self.class_filter, conf=conf, iou=iou, tracker=tracker,
            count_unique=count_unique, fps=self.fps, max_dim=max_dim,
            skip_bad_frames=skip_bad_frames,
        )
        denorm_lines = analysis['lines']
        line_counts = analysis['line_counts']
        num_frames = analysis['num_frames']

        default_dir = entry_direction or self.entry_direction
        per_line_dir = {idx: (line_directions or {}).get(idx, default_dir)
                        for idx in range(len(denorm_lines))}

        num_bins = int(np.ceil(num_frames / self.bin_size_frames)) if num_frames else 0
        roads = self._build_roads(line_counts, per_line_dir, num_bins)
        total_entries_series = self._sum_series(
            [r['time_series_entries'] for r in roads.values()])

        return {
            'video_path': video_path,
            'fps': self.fps,
            'bin_size_frames': self.bin_size_frames,
            'num_frames': num_frames,
            'num_bins': num_bins,
            'frame_size': analysis['frame_size'],
            'processed_frame_size': analysis.get('processed_frame_size'),
            'scale': analysis.get('scale', 1.0),
            'bin_times_seconds': [(i * self.bin_size_frames) / self.fps for i in range(num_bins)],
            'default_entry_direction': default_dir,
            'per_line_entry_direction': per_line_dir,
            'class_filter': self.class_filter,
            'lines': denorm_lines,
            'roads': roads,
            'junction_total_entries_series': total_entries_series,
            'junction_total_entries': sum(r['total_entries'] for r in roads.values()),
        }

    def _build_roads(self, line_counts, per_line_dir, num_bins):
        roads = {}
        for line_idx, summary in line_counts.items():
            entry_dir = per_line_dir.get(line_idx, self.entry_direction)
            opposite_dir = 'from_right' if entry_dir == 'from_left' else 'from_left'

            entry_events = summary[entry_dir]['events']
            exit_events = summary[opposite_dir]['events']

            entry_series = self._build_time_series(entry_events, num_bins)
            exit_series = self._build_time_series(exit_events, num_bins)
            entry_by_class = self._build_class_time_series(entry_events, num_bins)

            entry_track_ids = sorted({e['track_id'] for e in entry_events})
            exit_track_ids = sorted({e['track_id'] for e in exit_events})
            entry_class_counts = Counter(e['class_name'] for e in entry_events)
            exit_class_counts = Counter(e['class_name'] for e in exit_events)

            roads[line_idx] = {
                'line_index': line_idx,
                'line': summary['line'],
                'entry_direction': entry_dir,
                'exit_direction': opposite_dir,
                'total_entries': len(entry_events),
                'total_exits': len(exit_events),
                'unique_entry_tracks': len(entry_track_ids),
                'unique_exit_tracks': len(exit_track_ids),
                'entry_track_ids': entry_track_ids,
                'exit_track_ids': exit_track_ids,
                'entries_by_class': dict(entry_class_counts),
                'exits_by_class': dict(exit_class_counts),
                'time_series_entries': entry_series,
                'time_series_exits': exit_series,
                'time_series_by_class': entry_by_class,
                'entry_events': entry_events,
                'exit_events': exit_events,
                'crossings': self._build_crossings(entry_events + exit_events, line_idx),
            }
        return roads

    def build_road_record(self, road, road_index, fps, bin_size_frames, bin_seconds):
        time_series_entries = list(road.get('time_series_entries', []))
        return {
            'road_index': int(road_index),
            'fps': float(fps),
            'bin_size_frames': int(bin_size_frames),
            'bin_seconds': float(bin_seconds),
            'num_bins': len(time_series_entries),
            'entry_direction': road.get('entry_direction'),
            'exit_direction': road.get('exit_direction'),
            'time_series_entries': time_series_entries,
            'time_series_exits': list(road.get('time_series_exits', [])),
            'total_entries': int(road.get('total_entries', 0)),
            'total_exits': int(road.get('total_exits', 0)),
            'unique_entry_tracks': int(road.get('unique_entry_tracks', 0)),
            'unique_exit_tracks': int(road.get('unique_exit_tracks', 0)),
            'entries_by_class': dict(road.get('entries_by_class', {})),
            'exits_by_class': dict(road.get('exits_by_class', {})),
            'line': road.get('line'),
            'entry_events': list(road.get('entry_events', [])),
            'exit_events': list(road.get('exit_events', [])),
            'crossings': list(road.get('crossings', [])),
        }

    def build_export_record(self, video_path, sequence_key, fps, bin_size_frames,
                            bin_seconds, num_bins, roads):
        roads_out = {str(road_idx): self.build_road_record(road, road_idx, fps,
                                                           bin_size_frames, bin_seconds)
                     for road_idx, road in roads.items()}
        return {
            'video': video_path,
            'sequence_key': sequence_key,
            'fps': float(fps),
            'bin_size_frames': int(bin_size_frames),
            'bin_seconds': float(bin_seconds),
            'num_bins': int(num_bins),
            'num_roads': len(roads_out),
            'roads': roads_out,
        }

    def export_flow_to_dict(self, flow, video_path=None, sequence_key=None,
                            bin_seconds=None):
        bin_size_frames = flow.get('bin_size_frames', self.bin_size_frames)
        fps = flow.get('fps', self.fps)
        if bin_seconds is None:
            bin_seconds = bin_size_frames / fps if fps else 1.0
        return self.build_export_record(
            video_path=video_path or flow.get('video_path', ''),
            sequence_key=sequence_key, fps=fps,
            bin_size_frames=bin_size_frames, bin_seconds=bin_seconds,
            num_bins=flow.get('num_bins', 0), roads=flow.get('roads', {}),
        )

    def save_flow_to_json(self, flow, output_path, video_path=None,
                          sequence_key=None, bin_seconds=None):
        record = self.export_flow_to_dict(flow, video_path, sequence_key, bin_seconds)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(record, f, indent=2)
        return record

    def cumulative_series(self, flow_result):
        return {idx: list(np.cumsum(r['time_series_entries']).astype(int))
                for idx, r in flow_result['roads'].items()}

    def rate_per_minute(self, flow_result):
        bin_seconds = flow_result.get('bin_size_frames', self.bin_size_frames) / \
                      flow_result.get('fps', self.fps)
        scale = 60.0 / bin_seconds if bin_seconds > 0 else 0.0
        return {idx: [c * scale for c in r['time_series_entries']]
                for idx, r in flow_result['roads'].items()}

    def smoothed_series(self, flow_result, window_bins=5):
        window_bins = max(1, int(window_bins))
        out = {}
        for idx, road in flow_result['roads'].items():
            arr = np.array(road['time_series_entries'], dtype=np.float64)
            if len(arr) == 0:
                out[idx] = []
                continue
            kernel = np.ones(window_bins) / window_bins
            out[idx] = np.convolve(arr, kernel, mode='same').tolist()
        return out

    def summarize(self, flow_result):
        fps = flow_result.get('fps', self.fps)
        bin_size = flow_result.get('bin_size_frames', self.bin_size_frames)
        bin_seconds = bin_size / fps if fps else 0.0
        duration = flow_result['num_frames'] / fps if fps else 0.0
        per_road = {}
        for idx, road in flow_result['roads'].items():
            entries = road['total_entries']
            per_road[idx] = {
                'total_entries': entries,
                'unique_entries': road['unique_entry_tracks'],
                'entry_direction': road['entry_direction'],
                'avg_per_minute': (entries / duration * 60.0) if duration > 0 else 0.0,
                'peak_bin_count': max(road['time_series_entries']) if road['time_series_entries'] else 0,
                'entries_by_class': road['entries_by_class'],
            }
        return {
            'duration_seconds': duration,
            'bin_seconds': bin_seconds,
            'per_road': per_road,
            'junction_total': flow_result['junction_total_entries'],
            'junction_avg_per_minute': (
                flow_result['junction_total_entries'] / duration * 60.0
                if duration > 0 else 0.0
            ),
        }

    def _build_time_series(self, events, num_bins):
        series = [0] * num_bins
        for ev in events:
            bin_idx = (ev['frame'] - 1) // self.bin_size_frames
            if 0 <= bin_idx < num_bins:
                series[bin_idx] += 1
        return series

    def _build_class_time_series(self, events, num_bins):
        per_class = defaultdict(lambda: [0] * num_bins)
        for ev in events:
            bin_idx = (ev['frame'] - 1) // self.bin_size_frames
            if 0 <= bin_idx < num_bins:
                per_class[ev['class_name']][bin_idx] += 1
        return {k: list(v) for k, v in per_class.items()}

    @staticmethod
    def _build_crossings(events, line_index):
        return [{
            'time': float(ev.get('time_seconds', 0.0)),
            'vehicle_id': int(ev.get('track_id', -1)),
            'direction': ev.get('direction'),
            'line_index': int(line_index),
            'frame': int(ev.get('frame', 0)),
            'class_name': ev.get('class_name'),
        } for ev in events]

    @staticmethod
    def _sum_series(series_list):
        if not series_list:
            return []
        max_len = max(len(s) for s in series_list)
        out = [0] * max_len
        for s in series_list:
            for i, v in enumerate(s):
                out[i] += v
        return out

    @staticmethod
    def _infer_num_frames(results, line_counts):
        n_results = len(results) if results else 0
        max_event_frame = 0
        for summary in line_counts.values():
            for ev in summary['events']:
                if ev['frame'] > max_event_frame:
                    max_event_frame = ev['frame']
        return max(n_results, max_event_frame)