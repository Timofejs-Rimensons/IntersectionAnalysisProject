import os
import sys

os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import glob
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from services.TrackingModel import TrackingModel
from services.TrafficFlowEstimator import TrafficFlowEstimator
from services.FlowPredictionModel import FlowPredictionModel
from config.Config import Config


VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def find_videos(videos_dir):
    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    paths = []
    for f in sorted(os.listdir(videos_dir)):
        if f.lower().endswith(VIDEO_EXTS):
            paths.append(os.path.join(videos_dir, f))
    if not paths:
        raise RuntimeError(f"No video files found in {videos_dir}")
    return paths


def lookup_lines_for_video(boundary_data, video_path):
    seqs = boundary_data.get('sequences', boundary_data)
    if not isinstance(seqs, dict):
        raise ValueError("boundaries JSON must be a dict")

    stem = Path(video_path).stem
    name = Path(video_path).name

    for key in (stem, name):
        if key in seqs:
            entry = seqs[key]
            return entry.get('lines', []), bool(entry.get('normalized', False)), key

    for key, entry in seqs.items():
        if isinstance(entry, dict) and key in stem:
            return entry.get('lines', []), bool(entry.get('normalized', False)), key

    if len(seqs) == 1:
        only_key = next(iter(seqs))
        entry = seqs[only_key]
        return entry.get('lines', []), bool(entry.get('normalized', False)), only_key

    return None, None, None


def probe_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, 0, 0, 0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    if (fps <= 0 or fps > 1000) and n_frames > 0:
        cap = cv2.VideoCapture(video_path)
        try:
            t0 = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_frames - 1))
            cap.read()
            t1 = cap.get(cv2.CAP_PROP_POS_MSEC)
            duration_s = (t1 - t0) / 1000.0
            if duration_s > 0:
                fps = (n_frames - 1) / duration_s
        finally:
            cap.release()

    return fps, n_frames, w, h


def _scale_lines(lines, scale):
    if scale == 1.0:
        return [((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))
                for p1, p2 in lines]
    return [((int(p1[0] * scale), int(p1[1] * scale)),
             (int(p2[0] * scale), int(p2[1] * scale)))
            for p1, p2 in lines]


def estimate_flow_with_progress(estimator, tracking_model, video_path, lines,
                                normalized, conf, iou, fps, n_frames,
                                bin_size_frames, max_dim=1920,
                                skip_bad_frames=True, cleanup_every=200):
    import gc
    import torch

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if normalized:
        denorm_lines_orig = [
            ((int(p1[0] * img_w), int(p1[1] * img_h)),
             (int(p2[0] * img_w), int(p2[1] * img_h)))
            for p1, p2 in lines
        ]
    else:
        denorm_lines_orig = [
            ((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))
            for p1, p2 in lines
        ]

    analysis = tracking_model.analyze_crossings_from_video(
        video_path=video_path,
        lines=denorm_lines_orig,
        normalized=False,
        class_filter=estimator.class_filter,
        conf=conf,
        iou=iou,
        tracker='bytetrack.yaml',
        count_unique=False,
        fps=fps,
        max_dim=max_dim,
        skip_bad_frames=skip_bad_frames,
    )

    line_counts = analysis['line_counts']
    num_frames = analysis['num_frames']
    num_bins = (num_frames + bin_size_frames - 1) // bin_size_frames
    proc_w, proc_h = analysis.get('processed_frame_size', (img_w, img_h))
    scale = analysis.get('scale', 1.0)
    bad_frames = analysis.get('robust_meta', {}).get('bad_frames', 0)

    default_dir = estimator.entry_direction
    per_line_dir = {idx: default_dir for idx in range(len(denorm_lines_orig))}

    saved_bin_size = estimator.bin_size_frames
    estimator.bin_size_frames = bin_size_frames
    try:
        roads = {}
        for line_idx, summary in line_counts.items():
            entry_dir = per_line_dir[line_idx]
            opposite_dir = 'from_right' if entry_dir == 'from_left' else 'from_left'

            entry_events = summary[entry_dir]['events']
            exit_events = summary[opposite_dir]['events']

            entry_series = estimator._build_time_series(entry_events, num_bins)
            exit_series = estimator._build_time_series(exit_events, num_bins)
            entry_by_class = estimator._build_class_time_series(entry_events, num_bins)

            entry_track_ids = sorted({e['track_id'] for e in entry_events})
            exit_track_ids = sorted({e['track_id'] for e in exit_events})
            entry_class_counts = Counter(e['class_name'] for e in entry_events)
            exit_class_counts = Counter(e['class_name'] for e in exit_events)

            crossings = TrafficFlowEstimator._build_crossings(
                entry_events + exit_events, line_idx
            )

            roads[line_idx] = {
                'line_index': line_idx,
                'line': denorm_lines_orig[line_idx],
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
                'crossings': crossings,
            }
    finally:
        estimator.bin_size_frames = saved_bin_size

    return {
        'video_path': video_path,
        'fps': fps,
        'bin_size_frames': bin_size_frames,
        'num_frames': num_frames,
        'num_bins': num_bins,
        'frame_size': (img_w, img_h),
        'processed_frame_size': (proc_w, proc_h),
        'scale': scale,
        'bad_frames': bad_frames,
        'lines': denorm_lines_orig,
        'roads': roads,
    }


def extract_time_series(videos_dir, lines_json, tracking_weights, series_out,
                        config, conf=0.5, iou=0.45, fps=None,
                        bin_seconds=1.0, vehicle_classes=None,
                        entry_direction='from_left', max_dim=1920,
                        skip_bad_frames=True):
    print("\n" + "=" * 70)
    print("EXTRACTING TIME SERIES FROM VIDEOS")
    print("=" * 70)
    os.makedirs(series_out, exist_ok=True)

    with open(lines_json, 'r') as f:
        boundary_data = json.load(f)

    print(f"[1/3] Loading tracking model: {tracking_weights}")
    tracking_model = TrackingModel(
        config=config,
        checkpoint_path=tracking_weights,
        task='track',
        verbose=False,
    )
    print(f"      Classes: {tracking_model.classes}")

    print(f"[2/3] FPS auto-detected per video, bin size = {bin_seconds:.2f}s, "
          f"classes={vehicle_classes}")
    estimator = TrafficFlowEstimator(
        tracking_model=tracking_model,
        entry_direction=entry_direction,
        fps=30.0,
        bin_size_frames=1,
        class_filter=vehicle_classes,
    )

    print(f"[3/3] Discovering videos in: {videos_dir}")
    video_paths = find_videos(videos_dir)
    print(f"      Found {len(video_paths)} video(s)")

    extracted = []
    skipped = 0
    failed = 0

    overall_pbar = tqdm(
        total=len(video_paths),
        desc="Overall",
        unit="video",
        position=0,
        leave=True,
    )

    for v_idx, vp in enumerate(video_paths, start=1):
        v_name = os.path.basename(vp)
        overall_pbar.set_description(f"Overall [{v_idx}/{len(video_paths)}]")

        tqdm.write(f"\n[Video {v_idx}/{len(video_paths)}] {v_name}")

        v_fps, n_frames, v_w, v_h = probe_video(vp)
        if v_fps <= 0 or v_fps > 1000:
            fallback = fps if fps and fps > 0 else 30.0
            tqdm.write(f"  [WARN] Invalid FPS reported, using fallback {fallback}")
            v_fps = fallback

        duration = n_frames / v_fps if v_fps else 0
        tqdm.write(f"  Frames: {n_frames}  Resolution: {v_w}x{v_h}  "
                   f"FPS (detected): {v_fps:.3f}  Duration: {duration:.1f}s")

        bin_size_frames = max(1, int(round(bin_seconds * v_fps)))
        actual_bin_seconds = bin_size_frames / v_fps
        tqdm.write(f"  Binning: {bin_size_frames} frames/bin "
                   f"(~{actual_bin_seconds:.3f}s/bin)")

        estimator.fps = float(v_fps)
        estimator.bin_size_frames = bin_size_frames

        lines_data, normalized, resolved_key = lookup_lines_for_video(boundary_data, vp)
        if not lines_data:
            tqdm.write(f"  [SKIP] No lines defined in boundary JSON")
            skipped += 1
            overall_pbar.update(1)
            continue
        tqdm.write(f"  Lines: {len(lines_data)} (normalized={normalized}, "
                   f"key='{resolved_key}')")

        try:
            flow = estimate_flow_with_progress(
                estimator=estimator,
                tracking_model=tracking_model,
                video_path=vp,
                lines=lines_data,
                normalized=normalized,
                conf=conf,
                iou=iou,
                fps=v_fps,
                n_frames=n_frames,
                bin_size_frames=bin_size_frames,
                max_dim=max_dim,
                skip_bad_frames=skip_bad_frames,
            )
        except Exception as e:
            tqdm.write(f"  [FAIL] {type(e).__name__}: {e}")
            failed += 1
            overall_pbar.update(1)
            continue

        n_roads = len(flow['roads'])
        total_entries = sum(r['total_entries'] for r in flow['roads'].values())
        tqdm.write(f"  Done: {n_roads} road(s), {total_entries} total entries")

        stem = resolved_key or Path(vp).stem
        out_data = estimator.build_export_record(
            video_path=vp,
            sequence_key=resolved_key,
            fps=flow['fps'],
            bin_size_frames=flow['bin_size_frames'],
            bin_seconds=actual_bin_seconds,
            num_bins=flow['num_bins'],
            roads=flow['roads'],
        )

        out_path = os.path.join(series_out, f"{stem}.json")
        with open(out_path, 'w') as f:
            json.dump(out_data, f, indent=2)
        extracted.append(out_path)

        tqdm.write(f"  [OK] Saved {n_roads} road series for '{stem}'")
        overall_pbar.update(1)
        overall_pbar.set_postfix(ok=len(extracted), skip=skipped, fail=failed)

    overall_pbar.close()

    index_path = os.path.join(series_out, 'series_index.json')
    with open(index_path, 'w') as f:
        json.dump({
            'num_files': len(extracted),
            'files': extracted,
            'fps_mode': 'auto_per_video',
            'bin_seconds': bin_seconds,
            'entry_direction': entry_direction,
            'class_filter': vehicle_classes,
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"  Videos processed: {len(video_paths)}")
    print(f"    OK:      {len(video_paths) - skipped - failed}")
    print(f"    Skipped: {skipped}  (no lines)")
    print(f"    Failed:  {failed}")
    print(f"  Road series written: {len(extracted)}")
    print(f"  Output dir: {series_out}")
    print(f"  Index:      {index_path}")
    return extracted


def load_series_from_dir(series_dir, series_key='time_series_entries',
                         min_length=None):
    if not os.path.isdir(series_dir):
        raise FileNotFoundError(f"Series directory not found: {series_dir}")

    files = sorted(glob.glob(os.path.join(series_dir, '*.json')))
    files = [f for f in files if not f.endswith('series_index.json')]
    if not files:
        raise RuntimeError(f"No series JSON files in {series_dir}")

    series_list = []
    file_map = []
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
        except Exception:
            continue
        s = data.get(series_key)
        if not s:
            continue
        if min_length is not None and len(s) < min_length:
            continue
        series_list.append(s)
        file_map.append(f)

    if not series_list:
        raise RuntimeError("No usable series loaded")

    print(f"Loaded {len(series_list)} series from {series_dir}")
    return series_list, file_map


def load_intersection_series_from_dir(model, series_dir):
    if not os.path.isdir(series_dir):
        raise FileNotFoundError(f"Series directory not found: {series_dir}")

    print(f"Loading intersection series from {series_dir}")
    series_list, meta_list = model.load_intersection_series_from_dir(series_dir)
    if not series_list:
        raise RuntimeError("No usable intersection series loaded")
    print(f"Loaded {len(series_list)} intersection series from {series_dir}")
    return series_list, meta_list


def _pad_series_to_length(series, target_length):
    if series is None:
        return None
    arr = np.asarray(series, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if len(arr) >= target_length:
        return arr
    pad_len = target_length - len(arr)
    pad_shape = (pad_len, arr.shape[1])
    pad_values = np.zeros(pad_shape, dtype=np.float32)
    return np.vstack([pad_values, arr])


def split_train_val(series_list, meta_list=None, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    if meta_list is not None:
        groups = {}
        for idx, meta in enumerate(meta_list):
            group_key = meta.get('video_id') or meta.get('sequence_key') or f'series_{idx}'
            groups.setdefault(group_key, []).append(idx)

        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        n_val_groups = max(1, int(len(group_keys) * val_ratio)) if len(group_keys) > 1 else 0
        val_groups = set(group_keys[:n_val_groups])

        train_indices = []
        val_indices = []
        for group in group_keys:
            for idx in groups[group]:
                (val_indices if group in val_groups else train_indices).append(idx)

        train = [series_list[i] for i in train_indices]
        val = [series_list[i] for i in val_indices]
        return train, val

    indices = np.arange(len(series_list))
    rng.shuffle(indices)
    n_val = max(1, int(len(series_list) * val_ratio)) if len(series_list) > 1 else 0
    val_idx = set(indices[:n_val].tolist())
    train, val = [], []
    for i, s in enumerate(series_list):
        (val if i in val_idx else train).append(s)
    return train, val


def plot_history(history, run_dir):
    if not history.get('train_loss'):
        return
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_loss'], label='train', color='steelblue')
    if any(not np.isnan(v) for v in history.get('val_loss', [])):
        axes[0].plot(epochs, history['val_loss'], label='val', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_mae'], label='train', color='steelblue')
    if any(not np.isnan(v) for v in history.get('val_mae', [])):
        axes[1].plot(epochs, history['val_mae'], label='val', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (normalized)')
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(run_dir, 'history.png')
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"History plot: {out_path}")


def find_latest_run(weights_dir='weights', suffix='_flow'):
    if not os.path.exists(weights_dir):
        return None
    runs = sorted(
        [d for d in os.listdir(weights_dir)
         if os.path.isdir(os.path.join(weights_dir, d)) and suffix in d],
        reverse=True,
    )
    return os.path.join(weights_dir, runs[0]) if runs else None


def train_predictor(config, series_dir, resume_from=None, val_ratio=0.2,
                    batch_size=32, stride=1, verbose=False):
    model_type = getattr(config, 'pred_model_type', 'arima')
    print("\n" + "=" * 70)
    print(f"TRAINING {model_type.upper()} FOR TRAFFIC FLOW PREDICTION")
    print("=" * 70)

    run_dir = config.get_run_directory(task='flow', resume_from=resume_from)
    print(f"\nRun directory: {run_dir}\n", flush=True)

    model = FlowPredictionModel(config, verbose=verbose)
    series_list, meta_list = load_intersection_series_from_dir(model, series_dir)
    min_len = getattr(config, 'pred_input_window', 30) + getattr(config, 'pred_horizon', 1)
    padded_series = []
    padded_meta = []
    padded_count = 0
    for s, m in zip(series_list, meta_list):
        if len(s) < min_len:
            padded_count += 1
            original_length = len(s)
            s = _pad_series_to_length(s, min_len)
            m = dict(m)
            m['padded'] = True
            m['original_length'] = int(original_length)
        padded_series.append(s)
        padded_meta.append(m)
    series_list, meta_list = padded_series, padded_meta
    if padded_count > 0:
        print(f"Padded {padded_count} short intersection series to {min_len} timesteps.")

    train_series, val_series = split_train_val(
        list(series_list),
        meta_list=list(meta_list),
        val_ratio=val_ratio,
    )
    print(f"Train intersections: {len(train_series)} | Val intersections: {len(val_series)}")

    print(f"Initializing FlowPredictionModel with {model_type}...", flush=True)
    if model_type == 'linear':
        print("Fitting linear model...\n", flush=True)
        from services.FlowPredictionModel import TrafficFlowDataset
        train_ds = TrafficFlowDataset(
            series_list=train_series,
            input_window=getattr(config, 'pred_lookback', 5),
            horizon=getattr(config, 'pred_forecast_horizon', 1),
            stride=stride,
            normalize=True,
            augment=False,
        )
        X_train = []
        y_train = []
        for x, y in train_ds:
            X_train.append(x.numpy())
            y_train.append(y.numpy())
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        model.model.fit(X_train, y_train)
        model.save(os.path.join(run_dir, 'linear_model.pt'))
        print("Linear model fitted and saved.")
    elif model_type == 'arima':
        print("ARIMA forecasting model selected; no deep training phase required.\n", flush=True)
        model.save(os.path.join(run_dir, 'arima_model.pt'))
        print("ARIMA model metadata saved.")
    else:
        print("Starting training...\n", flush=True)
        try:
            model.train(
                train_series=train_series,
                val_series=val_series if val_series else None,
                run_dir=run_dir,
                batch_size=batch_size,
                stride=stride,
                num_workers=0,
                resume_from_last=bool(resume_from),
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user", flush=True)
            raise
        except Exception as e:
            print(f"\nTraining failed: {e}", flush=True)
            raise

        model.save_training_metrics(run_dir)
        plot_history(model.history, run_dir)

    model.create_run_summary(run_dir, model_name=model_type.upper())
    print(f"\n{'=' * 70}")
    print("Training completed")
    print(f"Run directory: {run_dir}")
    if hasattr(model, 'best_val_loss'):
        print(f"Best val loss: {model.best_val_loss:.6f}")
    print(f"{'=' * 70}")

    return model, run_dir


def predict_from_series_source(config, series_file=None, series_dir=None,
                               weights='best.pt', run_dir=None, steps=30,
                               verbose=False):
    print("\n" + "=" * 70)
    print("TRAFFIC FLOW FORECAST")
    print("=" * 70)

    if run_dir is None:
        run_dir = find_latest_run('weights', '_flow')
        if run_dir is None:
            raise ValueError("No flow training runs found in weights/")

    model_type = getattr(config, 'pred_model_type', 'arima')
    model_path = None
    if model_type == 'lstm':
        model_path = os.path.join(run_dir, weights)
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found: {model_path}")
        print(f"Using model: {model_path}")
    else:
        if weights and run_dir:
            candidate = os.path.join(run_dir, weights)
            if os.path.exists(candidate):
                model_path = candidate
                print(f"Loading saved predictor metadata: {model_path}")

    model = FlowPredictionModel(config, checkpoint_path=model_path, verbose=verbose)

    if series_dir:
        if not os.path.isdir(series_dir):
            raise FileNotFoundError(f"Series directory not found: {series_dir}")
        series_list, meta_list = model.load_intersection_series_from_dir(series_dir)
        if not series_list:
            raise RuntimeError(f"No usable intersection series loaded from {series_dir}")

        outputs = []
        for idx, (series, meta) in enumerate(zip(series_list, meta_list)):
            forecast = model.predict(series, steps=steps)
            output_dir = os.path.join(run_dir, f'inference_{meta.get("video_id", idx)}')
            os.makedirs(output_dir, exist_ok=True)
            forecast_path = os.path.join(output_dir, 'forecast.json')
            bin_seconds = meta.get('bin_seconds', getattr(model, 'bin_seconds', 1.0))
            forecast_data = {
                'source': series_dir,
                'video_id': meta.get('video_id'),
                'history_length': len(series),
                'forecast_steps': steps,
                'bin_seconds': bin_seconds,
                'forecast': forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast),
            }
            if isinstance(forecast, np.ndarray) and forecast.ndim == 2:
                flow_preds = [list(row) for row in forecast]
                per_road = {
                    f'road_{road_idx}': [row[road_idx] for row in flow_preds]
                    for road_idx in range(forecast.shape[1])
                }
                forecast_data['forecast_per_road'] = per_road
                forecast_data['forecast'] = flow_preds
                predicted_crossings = _generate_predicted_crossings(
                    flow_preds, len(series), bin_seconds, fps=30.0, multi_road=True
                )
                forecast_data['predicted_crossings'] = predicted_crossings
            with open(forecast_path, 'w') as f:
                json.dump(forecast_data, f, indent=2)
            print(f"Forecast saved: {forecast_path}")
            outputs.append(forecast_path)
        return outputs

    if not series_file:
        raise ValueError("predict mode requires --series-file or --series-dir")

    with open(series_file, 'r') as f:
        data = json.load(f)
    series = data.get('time_series_entries')
    if not series:
        raise ValueError(f"No 'time_series_entries' in {series_file}")
    print(f"Input series length: {len(series)}")

    forecast = model.predict(series, steps=steps)

    fps_val = data.get('fps', 30.0)
    bin_size = data.get('bin_size_frames', 1)
    bin_seconds = data.get('bin_seconds', bin_size / fps_val if fps_val > 0 else 1.0)

    output_dir = os.path.join(run_dir, f'inference_{Path(series_file).stem}')
    os.makedirs(output_dir, exist_ok=True)

    forecast_path = os.path.join(output_dir, 'forecast.json')
    forecast_data = {
        'source': series_file,
        'history': series,
        'forecast': forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast),
        'steps': steps,
        'bin_seconds': bin_seconds,
        'fps': fps_val,
    }
    if isinstance(forecast, np.ndarray) and forecast.ndim == 1:
        road_idx = data.get('road_index', 0)
        predicted_crossings = _generate_predicted_crossings(
            [[float(v)] for v in forecast], len(series), bin_seconds,
            fps=fps_val, multi_road=False, base_road_idx=road_idx
        )
        forecast_data['predicted_crossings'] = predicted_crossings
    with open(forecast_path, 'w') as f:
        json.dump(forecast_data, f, indent=2)
    print(f"Forecast saved: {forecast_path}")

    times_hist = np.arange(len(series)) * bin_seconds
    times_fut = (np.arange(steps) + len(series)) * bin_seconds

    plt.figure(figsize=(12, 5))
    plt.plot(times_hist, series, label='History', color='steelblue')
    plt.plot(times_fut, forecast, label='Forecast', color='red',
             linestyle='--', marker='o', markersize=3)
    plt.axvline(times_hist[-1], color='black', linewidth=0.6, linestyle=':')
    plt.xlabel('Time (s)')
    plt.ylabel('Vehicles per bin')
    plt.title(f'Flow forecast - {Path(series_file).stem}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(output_dir, 'forecast.png')
    plt.savefig(plot_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Forecast plot: {plot_path}")

    print(f"\nNext {steps} bins (each ~{bin_seconds:.2f}s):")
    for i, v in enumerate(forecast[:min(20, steps)]):
        print(f"  t+{i + 1:>3}: {float(v):.2f} veh/bin")
    if steps > 20:
        print(f"  ... ({steps - 20} more)")

    return [forecast_path]


def _generate_predicted_crossings(flow_preds, history_len, bin_seconds,
                                  fps=30.0, multi_road=True,
                                  base_road_idx=0):
    predicted_crossings = []
    vehicle_id_counter = 1
    rng = np.random.default_rng(42)
    for step_idx, row in enumerate(flow_preds):
        bin_start_time = (history_len + step_idx) * bin_seconds
        if not isinstance(row, (list, tuple)):
            row = [row]
        for road_offset, count in enumerate(row):
            road_idx = base_road_idx + road_offset if not multi_road else road_offset
            count_int = max(0, int(round(count)))
            for _ in range(count_int):
                sub_time = bin_start_time + rng.uniform(0, bin_seconds)
                predicted_crossings.append({
                    'time': float(sub_time),
                    'vehicle_id': vehicle_id_counter,
                    'direction': 'from_left',
                    'line_index': road_idx,
                    'frame': int(sub_time * fps),
                    'class_name': 'predicted_vehicle',
                })
                vehicle_id_counter += 1
    return predicted_crossings


def main():
    parser = argparse.ArgumentParser(description='Traffic Flow Prediction')
    parser.add_argument('--mode', type=str,
                        choices=['extract', 'train', 'predict', 'generate-synthetic'],
                        default='train')

    parser.add_argument('--videos-dir', type=str)
    parser.add_argument('--lines-json', type=str)
    parser.add_argument('--tracking-weights', type=str)
    parser.add_argument('--series-out', type=str, default='./datasets/flow_series')
    parser.add_argument('--series-dir', type=str)
    parser.add_argument('--series-file', type=str)

    parser.add_argument('--fps', type=float, default=None,
                        help='Fallback FPS if video metadata is missing/invalid')
    parser.add_argument('--bin-seconds', type=float, default=1.0,
                        help='Time-series bin size in seconds')
    parser.add_argument('--vehicle-classes', type=str, default='car,truck,bus')
    parser.add_argument('--entry-direction', type=str, default='from_left',
                        choices=['from_left', 'from_right'])
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--max-dim', type=int, default=1920,
                        help='Downscale frames so the largest side <= this')
    parser.add_argument('--no-skip-bad-frames', action='store_true',
                        help='Stop on decode errors instead of skipping')

    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--resume', type=str)

    parser.add_argument('--weights', type=str, default='best.pt')
    parser.add_argument('--run-dir', type=str)
    parser.add_argument('--steps', type=int, default=30)

    parser.add_argument('--synthetic-rules', type=str,
                        help='JSON file with synthetic generation rules')
    parser.add_argument('--synthetic-count', type=int, default=10,
                        help='Number of synthetic intersections to generate')
    parser.add_argument('--synthetic-bins', type=int, default=60,
                        help='Number of time bins per synthetic intersection')

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    vehicle_classes = [c.strip() for c in args.vehicle_classes.split(',')
                       if c.strip()] if args.vehicle_classes else None

    if args.mode == 'extract':
        if not (args.videos_dir and args.lines_json and args.tracking_weights):
            raise ValueError(
                "extract mode requires --videos-dir, --lines-json, "
                "--tracking-weights"
            )
        extract_time_series(
            videos_dir=args.videos_dir,
            lines_json=args.lines_json,
            tracking_weights=args.tracking_weights,
            series_out=args.series_out,
            config=config,
            conf=args.conf,
            iou=args.iou,
            fps=args.fps,
            bin_seconds=args.bin_seconds,
            vehicle_classes=vehicle_classes,
            entry_direction=args.entry_direction,
            max_dim=args.max_dim,
            skip_bad_frames=not args.no_skip_bad_frames,
        )
        print(f"\nTrain with:")
        print(f"  python flow_predictor.py --mode train --series-dir {args.series_out}")
        return

    if args.mode == 'generate-synthetic':
        generate_synthetic_data(
            config=config,
            rules_file=args.synthetic_rules,
            output_dir=args.series_out,
            count=args.synthetic_count,
            num_bins=args.synthetic_bins,
            fps=args.fps or 30.0,
            bin_seconds=args.bin_seconds,
            entry_direction=args.entry_direction,
            vehicle_classes=vehicle_classes,
        )
        return

    if args.mode == 'train':
        series_dir = args.series_dir or args.series_out
        if not os.path.isdir(series_dir):
            if args.videos_dir and args.lines_json and args.tracking_weights:
                print(f"series_dir {series_dir} not found, running extraction first")
                extract_time_series(
                    videos_dir=args.videos_dir,
                    lines_json=args.lines_json,
                    tracking_weights=args.tracking_weights,
                    series_out=args.series_out,
                    config=config,
                    conf=args.conf,
                    iou=args.iou,
                    fps=args.fps,
                    bin_seconds=args.bin_seconds,
                    vehicle_classes=vehicle_classes,
                    entry_direction=args.entry_direction,
                    max_dim=args.max_dim,
                    skip_bad_frames=not args.no_skip_bad_frames,
                )
                series_dir = args.series_out
            else:
                raise ValueError(
                    f"--series-dir {series_dir} not found and no extraction "
                    "args provided"
                )

        train_predictor(
            config=config,
            series_dir=series_dir,
            resume_from=args.resume,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            stride=args.stride,
            verbose=args.verbose,
        )
        return

    if args.mode == 'predict':
        if not args.series_file and not args.series_dir:
            raise ValueError("predict mode requires --series-file or --series-dir")
        predict_from_series_source(
            config=config,
            series_file=args.series_file,
            series_dir=args.series_dir,
            weights=args.weights,
            run_dir=args.run_dir,
            steps=args.steps,
            verbose=args.verbose,
        )
        return


def generate_synthetic_data(config, rules_file=None, output_dir='./datasets/flow_series',
                            count=10, num_bins=60, fps=30.0, bin_seconds=1.0,
                            entry_direction='from_left', vehicle_classes=None):
    print("\n" + "=" * 70)
    print("GENERATING SYNTHETIC TRAFFIC FLOW DATA")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)

    estimator = TrafficFlowEstimator(
        tracking_model=None,
        entry_direction=entry_direction,
        fps=fps,
        bin_size_frames=max(1, int(round(bin_seconds * fps))),
        class_filter=vehicle_classes,
    )

    if rules_file and os.path.isfile(rules_file):
        with open(rules_file, 'r') as f:
            rules_data = json.load(f)

        if isinstance(rules_data, list):
            dataset_rules = rules_data
        elif isinstance(rules_data, dict) and 'intersections' in rules_data:
            dataset_rules = rules_data['intersections']
        elif isinstance(rules_data, dict) and 'roads' in rules_data:
            dataset_rules = [rules_data] * count
        else:
            dataset_rules = _build_default_dataset_rules(
                count=count, num_bins=num_bins, fps=fps,
                bin_seconds=bin_seconds, entry_direction=entry_direction,
                vehicle_classes=vehicle_classes,
            )
        print(f"Loaded {len(dataset_rules)} intersection specs from {rules_file}")
    else:
        dataset_rules = _build_default_dataset_rules(
            count=count, num_bins=num_bins, fps=fps,
            bin_seconds=bin_seconds, entry_direction=entry_direction,
            vehicle_classes=vehicle_classes,
        )
        print(f"Using default rules to generate {count} intersections")

    for spec in dataset_rules:
        spec.setdefault('num_bins', num_bins)
        spec.setdefault('fps', fps)
        spec.setdefault('bin_seconds', bin_seconds)

    files = estimator.generate_synthetic_dataset(
        dataset_rules=dataset_rules,
        output_dir=output_dir,
        seed=42,
    )

    index_path = os.path.join(output_dir, 'series_index.json')
    with open(index_path, 'w') as f:
        json.dump({
            'num_files': len(files),
            'files': files,
            'fps_mode': 'fixed',
            'bin_seconds': bin_seconds,
            'entry_direction': entry_direction,
            'class_filter': vehicle_classes,
            'synthetic': True,
        }, f, indent=2)

    print(f"\nGenerated {len(files)} synthetic intersection files")
    print(f"Output dir: {output_dir}")
    print(f"Index: {index_path}")
    return files


def _build_default_dataset_rules(count, num_bins, fps, bin_seconds,
                                 entry_direction, vehicle_classes):
    rng = np.random.default_rng(0)
    classes = vehicle_classes or ['car', 'truck', 'bus']
    rules = []
    for i in range(count):
        n_roads = rng.integers(2, 5)
        road_rules = []
        for r in range(n_roads):
            base_mean = float(rng.uniform(0.5, 4.0))
            base_std = float(rng.uniform(0.3, max(0.31, base_mean * 0.5)))
            exit_ratio = float(rng.uniform(0.6, 1.0))
            weights = rng.dirichlet(np.ones(len(classes)))
            class_distribution = {c: float(w) for c, w in zip(classes, weights)}
            line_x1 = int(rng.integers(50, 200))
            line_y1 = int(rng.integers(50, 300))
            line_x2 = int(rng.integers(300, 600))
            line_y2 = int(rng.integers(50, 300))
            road_rules.append({
                'entry_mean': base_mean,
                'entry_std': base_std,
                'exit_mean': base_mean * exit_ratio,
                'exit_std': base_std * exit_ratio,
                'entry_direction': entry_direction,
                'class_distribution': class_distribution,
                'line': ((line_x1, line_y1), (line_x2, line_y2)),
            })
        rules.append({
            'sequence_key': f'synthetic_{i:04d}',
            'video_path': f'synthetic_{i:04d}.mp4',
            'num_bins': num_bins,
            'fps': fps,
            'bin_seconds': bin_seconds,
            'roads': road_rules,
        })
    return rules


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("TRAFFIC FLOW PREDICTION")
    print("=" * 70)

    if len(sys.argv) == 1:
        print("\nUSAGE")
        print("=" * 70)
        print("\n1. EXTRACT TIME SERIES FROM INTERSECTION VIDEOS:")
        print("   python flow_predictor.py --mode extract \\")
        print("     --videos-dir ./datasets/junction_videos \\")
        print("     --lines-json ./datasets/crossection_boundaries.json \\")
        print("     --tracking-weights ./weights/<track-run>/best.pt \\")
        print("     --series-out ./datasets/flow_series \\")
        print("     --bin-seconds 1.0 \\")
        print("     --vehicle-classes car,truck,bus")

        print("\n2. TRAIN FROM EXTRACTED SERIES:")
        print("   python flow_predictor.py --mode train \\")
        print("     --series-dir ./datasets/flow_series \\")
        print("     --val-ratio 0.2 --batch-size 32 --verbose")

        print("\n3. EXTRACT + TRAIN IN ONE GO:")
        print("   python flow_predictor.py --mode train \\")
        print("     --videos-dir ./datasets/junction_videos \\")
        print("     --lines-json ./datasets/crossection_boundaries.json \\")
        print("     --tracking-weights ./weights/<track-run>/best.pt \\")
        print("     --series-out ./datasets/flow_series")

        print("\n4. RESUME TRAINING:")
        print("   python flow_predictor.py --mode train \\")
        print("     --series-dir ./datasets/flow_series \\")
        print("     --resume weights/<flow-run>/")

        print("\n5. FORECAST FOR ONE ROAD:")
        print("   python flow_predictor.py --mode predict \\")
        print("     --series-file ./datasets/flow_series/<seq>_road0.json \\")
        print("     --steps 30 --weights best.pt")

        print("\n6. GENERATE SYNTHETIC DATA:")
        print("   python flow_predictor.py --mode generate-synthetic \\")
        print("     --series-out ./datasets/flow_series \\")
        print("     --synthetic-count 20 --synthetic-bins 120 \\")
        print("     --bin-seconds 1.0 --vehicle-classes car,truck,bus")

        print("\n7. GENERATE SYNTHETIC FROM RULES FILE:")
        print("   python flow_predictor.py --mode generate-synthetic \\")
        print("     --synthetic-rules ./rules.json \\")
        print("     --series-out ./datasets/flow_series")
        print()
        sys.exit(0)

    main()