import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import glob
import math
from pathlib import Path

try:
    from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA
except ImportError:
    StatsmodelsARIMA = None


class TrafficFlowDataset(Dataset):
    def __init__(self, series_list, input_window=30, horizon=1, stride=1,
                 normalize=True, stats=None, augment=False,
                 augment_time_shift=False, time_shift_max_phase=1.0,
                 time_feature_count=0):
        self.input_window = int(input_window)
        self.horizon = int(horizon)
        self.stride = max(1, int(stride))
        self.normalize = normalize
        self.augment = augment
        self.augment_time_shift = augment_time_shift
        self.time_shift_max_phase = float(time_shift_max_phase)
        self.time_feature_count = int(time_feature_count)

        prepared = []
        for s in series_list:
            arr = np.asarray(s, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            prepared.append(arr)

        if not prepared:
            raise ValueError("series_list must contain at least one series")

        self.num_features = prepared[0].shape[1]
        for arr in prepared:
            if arr.shape[1] != self.num_features:
                raise ValueError(
                    f"All series must have the same feature count "
                    f"(got {arr.shape[1]} vs {self.num_features})"
                )

        if stats is not None:
            self.mean = np.asarray(stats['mean'], dtype=np.float32)
            self.std = np.asarray(stats['std'], dtype=np.float32)
        elif normalize:
            stacked = np.concatenate(prepared, axis=0)
            self.mean = stacked.mean(axis=0).astype(np.float32)
            self.std = stacked.std(axis=0).astype(np.float32)
            self.std[self.std < 1e-6] = 1.0
        else:
            self.mean = np.zeros(self.num_features, dtype=np.float32)
            self.std = np.ones(self.num_features, dtype=np.float32)

        self.samples = []
        for arr in prepared:
            if normalize:
                arr = (arr - self.mean) / self.std
            n = len(arr)
            last_start = n - self.input_window - self.horizon + 1
            for i in range(0, last_start, self.stride):
                x = arr[i: i + self.input_window]
                y = arr[i + self.input_window: i + self.input_window + self.horizon]
                self.samples.append((x, y))

        if not self.samples:
            raise ValueError(
                "No samples produced. Check input_window/horizon vs series length."
            )

    def __len__(self):
        return len(self.samples)

    def _apply_time_shift(self, x):
        if self.time_feature_count < 2 or x.shape[1] < self.time_feature_count:
            return x

        sin_idx = x.shape[1] - 2
        cos_idx = x.shape[1] - 1
        sin_vals = x[:, sin_idx].astype(np.float32)
        cos_vals = x[:, cos_idx].astype(np.float32)

        raw_sin = sin_vals * self.std[sin_idx] + self.mean[sin_idx]
        raw_cos = cos_vals * self.std[cos_idx] + self.mean[cos_idx]
        theta = np.arctan2(raw_sin, raw_cos)
        delta = np.random.uniform(-np.pi * self.time_shift_max_phase,
                                  np.pi * self.time_shift_max_phase)
        shifted_sin = np.sin(theta + delta).astype(np.float32)
        shifted_cos = np.cos(theta + delta).astype(np.float32)

        x_shifted = x.copy()
        x_shifted[:, sin_idx] = (shifted_sin - self.mean[sin_idx]) / self.std[sin_idx]
        x_shifted[:, cos_idx] = (shifted_cos - self.mean[cos_idx]) / self.std[cos_idx]
        return x_shifted

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.augment and self.augment_time_shift and self.time_feature_count >= 2:
            x = self._apply_time_shift(x)
        return torch.from_numpy(x), torch.from_numpy(y)

    def get_stats(self):
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}


class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 horizon=1, dropout=0.2, output_size=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.output_size = output_size or input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, self.output_size * horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        y = self.head(last)
        return y.view(x.size(0), self.horizon, self.output_size)


class LinearPredictor:
    def __init__(self, lookback=5, horizon=1, num_features=1):
        from sklearn.linear_model import LinearRegression
        self.lookback = lookback
        self.horizon = horizon
        self.num_features = num_features
        self.models = [LinearRegression() for _ in range(horizon * num_features)]

    def fit(self, X, y):
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        for i in range(y_flat.shape[1]):
            self.models[i].fit(X_flat, y_flat[:, i])

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        preds = np.array([model.predict(X_flat) for model in self.models]).T
        return preds.reshape(X.shape[0], self.horizon, self.num_features)


class ARIMAPredictor:
    def __init__(self, order=(1, 1, 0), seasonal_order=(0, 0, 0, 0),
                 trend='c', horizon=1, num_features=1, add_time_feature=False):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.horizon = int(horizon)
        self.num_features = int(num_features)
        self.add_time_feature = bool(add_time_feature)

    @staticmethod
    def _ensure_dependency():
        if StatsmodelsARIMA is None:
            raise ImportError(
                "statsmodels is required for ARIMA prediction. "
                "Install it with `pip install statsmodels`."
            )

    def _strip_time_features(self, arr):
        if self.add_time_feature and arr.ndim == 2 and arr.shape[1] > self.num_features:
            return arr[:, :self.num_features]
        return arr

    def predict(self, history, steps=None, clip_negative=True, round_int=False):
        self._ensure_dependency()
        steps = int(steps) if steps is not None else self.horizon
        arr = np.asarray(history, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arr = self._strip_time_features(arr)
        if arr.shape[1] != self.num_features:
            raise ValueError(
                f"ARIMA predictor expects {self.num_features} target feature(s), "
                f"got {arr.shape[1]}"
            )

        forecast = np.zeros((steps, self.num_features), dtype=np.float32)
        for feature_idx in range(self.num_features):
            series = arr[:, feature_idx].astype(np.float32)
            if len(series) < max(1, self.order[0] + self.order[1] + self.order[2]):
                raise ValueError(
                    f"ARIMA history length {len(series)} is too short for order {self.order}"
                )
            model = StatsmodelsARIMA(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
            )
            result = model.fit()
            preds = result.forecast(steps)
            forecast[:, feature_idx] = np.asarray(preds, dtype=np.float32)

        if clip_negative:
            forecast = np.maximum(forecast, 0.0)
        if round_int:
            forecast = np.round(forecast)

        return forecast.reshape(-1) if self.num_features == 1 else forecast


class FlowPredictionModel:
    def __init__(self, config, checkpoint_path=None, verbose=False):
        self.config = config
        self.device = torch.device(config.device)
        self.verbose = verbose

        self.num_lines = getattr(config, 'pred_num_lines', 4)
        self.include_exits = getattr(config, 'pred_include_exits', False)
        self.add_time_feature = getattr(config, 'pred_add_time_feature', True)
        self.time_period_seconds = getattr(config, 'pred_time_period_seconds', 60.0)
        self.time_shift_max_phase = getattr(config, 'pred_time_shift_max_phase', 1.0)
        self.early_stopping_patience = getattr(config, 'pred_early_stopping_patience', 10)
        self.early_stopping_min_delta = getattr(config, 'pred_early_stopping_min_delta', 1e-4)

        flow_features = self.num_lines * (2 if self.include_exits else 1)
        time_features = 2 if self.add_time_feature else 0
        default_input = flow_features + time_features
        default_output = flow_features

        self.input_size = getattr(config, 'pred_input_size', default_input)
        self.output_size = getattr(config, 'pred_output_size', default_output)
        self.hidden_size = getattr(config, 'pred_hidden_size', 128)
        self.num_layers = getattr(config, 'pred_num_layers', 2)
        self.horizon = getattr(config, 'pred_horizon', 1)
        self.input_window = getattr(config, 'pred_input_window', 30)
        self.dropout = getattr(config, 'pred_dropout', 0.2)

        self.bin_seconds = getattr(config, 'pred_bin_seconds', 1.0)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "lr": [],
        }
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.norm_stats = None

        self.model = self._build_model(checkpoint_path=checkpoint_path)

    def _build_model(self, checkpoint_path=None):
        model_type = getattr(self.config, 'pred_model_type', 'arima')
        if model_type == 'linear':
            self.model = LinearPredictor(
                lookback=self.input_window,
                horizon=self.horizon,
                num_features=self.output_size
            )
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    candidate = self._safe_load_checkpoint(checkpoint_path)
                    if isinstance(candidate, LinearPredictor):
                        self.model = candidate
                    else:
                        self.model = candidate
                    return self.model
                except Exception:
                    pass
            return self.model

        if model_type == 'arima':
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    candidate = self._safe_load_checkpoint(checkpoint_path)
                    if isinstance(candidate, ARIMAPredictor):
                        self.model = candidate
                        return candidate
                    self.model = candidate
                    return candidate
                except Exception:
                    pass
            self.model = ARIMAPredictor(
                order=getattr(self.config, 'pred_arima_order', (1, 0, 0)),
                seasonal_order=getattr(self.config, 'pred_arima_seasonal_order', (0, 0, 0, 0)),
                trend=getattr(self.config, 'pred_arima_trend', 'c'),
                horizon=self.horizon,
                num_features=self.output_size,
                add_time_feature=self.add_time_feature,
            )
            return self.model

        if checkpoint_path and os.path.exists(checkpoint_path):
            if self.verbose:
                print(f"Loading checkpoint from {checkpoint_path}")
            ckpt = self._safe_load_checkpoint(checkpoint_path)
            if isinstance(ckpt, dict) and 'arch' in ckpt:
                cfg = ckpt['arch']
                self.input_size = cfg.get('input_size', self.input_size)
                self.hidden_size = cfg.get('hidden_size', self.hidden_size)
                self.num_layers = cfg.get('num_layers', self.num_layers)
                self.horizon = cfg.get('horizon', self.horizon)
                self.input_window = cfg.get('input_window', self.input_window)
                self.dropout = cfg.get('dropout', self.dropout)
                self.output_size = cfg.get('output_size', self.output_size)
                self.num_lines = cfg.get('num_lines', self.num_lines)
                self.include_exits = cfg.get('include_exits', self.include_exits)
                self.add_time_feature = cfg.get('add_time_feature', self.add_time_feature)
                self.time_period_seconds = cfg.get('time_period_seconds', self.time_period_seconds)
                self.bin_seconds = cfg.get('bin_seconds', self.bin_seconds)
        else:
            ckpt = None

        model = TrafficLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            horizon=self.horizon,
            dropout=self.dropout,
            output_size=self.output_size,
        ).to(self.device)

        if ckpt is not None:
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                self.history = ckpt.get('history', self.history)
                self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
                self.start_epoch = ckpt.get('epoch', 0)
                self.norm_stats = ckpt.get('norm_stats', None)
            else:
                model.load_state_dict(ckpt)
        elif self.verbose:
            print("Initializing fresh LSTM model.")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.verbose:
            print(f"Parameters: {total_params:,} total | "
                  f"{trainable_params:,} trainable")
        if trainable_params == 0:
            raise RuntimeError("Model has 0 trainable parameters.")
        return model

    @staticmethod
    def extract_crossings(flow_result, video_id=None):
        out = []
        for line_idx, road in flow_result.get('roads', {}).items():
            for ev in road.get('entry_events', []):
                out.append({
                    'time': float(ev.get('time_seconds') or 0.0),
                    'frame': int(ev.get('frame', 0)),
                    'line_no': int(line_idx),
                    'vehicle_id': int(ev.get('track_id', -1)),
                    'class_name': ev.get('class_name'),
                    'direction': ev.get('direction', road.get('entry_direction')),
                    'video_id': video_id,
                })
        out.sort(key=lambda e: (e['time'], e['line_no']))
        return out

    @staticmethod
    def crossings_from_series_json(series_json_data, video_id=None):
        out = []
        line_no = series_json_data.get('road_index', 0)
        for ev in series_json_data.get('entry_events', []):
            out.append({
                'time': float(ev.get('time_seconds') or 0.0),
                'frame': int(ev.get('frame', 0)),
                'line_no': int(line_no),
                'vehicle_id': int(ev.get('track_id', -1)),
                'class_name': ev.get('class_name'),
                'direction': ev.get('direction'),
                'video_id': video_id,
            })
        out.sort(key=lambda e: (e['time'], e['line_no']))
        return out

    def _add_time_features(self, arr, bin_seconds, start_offset=0.0):
        if not self.add_time_feature:
            return arr
        n = arr.shape[0]
        t = (np.arange(n, dtype=np.float32) + start_offset) * bin_seconds
        period = max(self.time_period_seconds, 1e-3)
        sin_t = np.sin(2.0 * math.pi * t / period).astype(np.float32)
        cos_t = np.cos(2.0 * math.pi * t / period).astype(np.float32)
        time_feat = np.stack([sin_t, cos_t], axis=1)
        return np.concatenate([arr, time_feat], axis=1)

    def _resample_time_series(self, series, orig_bin_seconds, target_bin_seconds):
        if not series:
            return []
        if abs(orig_bin_seconds - target_bin_seconds) < 1e-6:
            return [float(v) for v in series]

        series = np.asarray(series, dtype=np.float32)
        total_time = len(series) * orig_bin_seconds
        num_bins = int(np.ceil(total_time / target_bin_seconds))
        out = np.zeros(num_bins, dtype=np.float32)

        for idx, value in enumerate(series):
            start = idx * orig_bin_seconds
            end = start + orig_bin_seconds
            first_bin = int(np.floor(start / target_bin_seconds))
            last_bin = int(np.floor((end - 1e-9) / target_bin_seconds))
            for b in range(first_bin, last_bin + 1):
                bin_start = b * target_bin_seconds
                bin_end = bin_start + target_bin_seconds
                overlap = max(0.0, min(end, bin_end) - max(start, bin_start))
                if overlap > 0:
                    out[b] += float(value) * (overlap / orig_bin_seconds)
        return out.tolist()

    def build_intersection_series(self, road_series_dict, bin_seconds=None):
        if bin_seconds is None:
            bin_seconds = self.bin_seconds

        line_indices = sorted(road_series_dict.keys())[:self.num_lines]
        if not line_indices:
            raise ValueError("No road series provided")

        resampled = {}
        for k in line_indices:
            road = road_series_dict[k]
            orig_bin_seconds = float(road.get('bin_seconds', bin_seconds))
            entries = self._resample_time_series(
                road.get('time_series_entries', []),
                orig_bin_seconds,
                bin_seconds,
            )
            exits = self._resample_time_series(
                road.get('time_series_exits', []),
                orig_bin_seconds,
                bin_seconds,
            ) if self.include_exits else []
            resampled[k] = {
                'entries': entries,
                'exits': exits,
            }

        max_len = max(len(resampled[k]['entries']) for k in line_indices)
        flow_features = self.num_lines * (2 if self.include_exits else 1)
        arr = np.zeros((max_len, flow_features), dtype=np.float32)

        for slot, k in enumerate(line_indices):
            entries = resampled[k]['entries']
            for t, v in enumerate(entries):
                arr[t, slot] = float(v)
            if self.include_exits:
                exits = resampled[k]['exits']
                for t, v in enumerate(exits):
                    arr[t, self.num_lines + slot] = float(v)

        return self._add_time_features(arr, bin_seconds)

    def load_intersection_series_from_dir(self, series_dir, group_by='video'):
        files = sorted(glob.glob(os.path.join(series_dir, '*.json')))
        files = [f for f in files if not f.endswith('series_index.json')]

        series_list = []
        meta_list = []
        for f in files:
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
            except Exception:
                continue
            if 'roads' not in data:
                continue
            bin_seconds = float(data.get('bin_seconds', self.bin_seconds))
            try:
                arr = self.build_intersection_series(data['roads'], bin_seconds=bin_seconds)
            except Exception:
                continue
            series_list.append(arr)
            meta_list.append({
                'video_id': data.get('sequence_key') or Path(data.get('video', f)).stem,
                'num_roads': data.get('num_roads', 0),
                'bin_seconds': bin_seconds,
                'length': arr.shape[0],
            })

        if not series_list:
            raise RuntimeError(f"No intersection series built from {series_dir}")
        return series_list, meta_list

    def _make_loaders(self, train_series, val_series=None, batch_size=32,
                      stride=1, num_workers=0):
        time_feature_count = 2 if self.add_time_feature else 0
        train_ds = TrafficFlowDataset(
            series_list=train_series,
            input_window=self.input_window,
            horizon=self.horizon,
            stride=stride,
            normalize=True,
            augment=self.config.pred_augment_time_shift,
            augment_time_shift=self.config.pred_augment_time_shift,
            time_shift_max_phase=self.config.pred_time_shift_max_phase,
            time_feature_count=time_feature_count,
        )
        self.norm_stats = train_ds.get_stats()

        val_ds = None
        if val_series:
            val_ds = TrafficFlowDataset(
                series_list=val_series,
                input_window=self.input_window,
                horizon=self.horizon,
                stride=1,
                normalize=True,
                stats=self.norm_stats,
                augment=False,
                augment_time_shift=False,
                time_feature_count=time_feature_count,
            )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=False,
        )
        val_loader = None
        if val_ds is not None and len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, drop_last=False,
            )
        return train_loader, val_loader

    def _output_to_input(self, y_step, time_step_idx, bin_seconds):
        if self.add_time_feature:
            t = time_step_idx * bin_seconds
            period = max(self.time_period_seconds, 1e-3)
            sin_t = math.sin(2.0 * math.pi * t / period)
            cos_t = math.cos(2.0 * math.pi * t / period)
            return np.concatenate([y_step,
                                   np.array([sin_t, cos_t], dtype=np.float32)])
        return y_step

    def _run_epoch(self, loader, criterion, optimizer=None):
        is_train = optimizer is not None
        self.model.train(is_train)

        total_loss = 0.0
        total_mae = 0.0
        total_count = 0

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if y.shape[-1] != self.output_size:
                y = y[..., :self.output_size]

            if is_train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                pred = self.model(x)
                loss = criterion(pred, y)
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_mae += (pred - y).abs().mean().item() * bs
            total_count += bs

        if total_count == 0:
            return 0.0, 0.0
        return total_loss / total_count, total_mae / total_count

    def train(self, train_series, val_series=None, run_dir=None,
              batch_size=32, stride=1, num_workers=0, resume_from_last=True):
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)

        if train_series and isinstance(train_series[0], np.ndarray):
            seen_features = train_series[0].shape[-1]
            if seen_features != self.input_size:
                if self.verbose:
                    print(f"Adjusting input_size {self.input_size} -> {seen_features}")
                self.input_size = seen_features
                if not self.add_time_feature:
                    self.output_size = seen_features
                else:
                    self.output_size = seen_features - 2
                self.model = TrafficLSTM(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    horizon=self.horizon,
                    dropout=self.dropout,
                    output_size=self.output_size,
                ).to(self.device)

        train_loader, val_loader = self._make_loaders(
            train_series=train_series,
            val_series=val_series,
            batch_size=batch_size,
            stride=stride,
            num_workers=num_workers,
        )

        if self.verbose:
            print(f"Train samples: {len(train_loader.dataset)} | "
                  f"Val samples: {len(val_loader.dataset) if val_loader else 0}")
            print(f"Input size: {self.input_size} | Output size: {self.output_size}")

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs,
            eta_min=self.config.lr * 0.01,
        )

        patience_counter = 0
        best_val_loss = self.best_val_loss
        last_ckpt = os.path.join(run_dir, 'last.pt') if run_dir else None
        best_ckpt = os.path.join(run_dir, 'best.pt') if run_dir else None

        if resume_from_last and last_ckpt and os.path.exists(last_ckpt):
            try:
                ckpt = self._safe_load_checkpoint(last_ckpt)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    self.model.load_state_dict(ckpt['model_state_dict'])
                    if 'optimizer_state_dict' in ckpt:
                        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    self.start_epoch = ckpt.get('epoch', 0)
                    self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
                    self.history = ckpt.get('history', self.history)
                    self.norm_stats = ckpt.get('norm_stats', self.norm_stats)
                    if self.verbose:
                        print(f"Resumed from epoch {self.start_epoch}")
            except Exception as e:
                print(f"Resume failed: {e}")

        epoch = self.start_epoch
        try:
            for epoch in range(self.start_epoch, self.config.num_epochs):
                lr_now = optimizer.param_groups[0]['lr']
                train_loss, train_mae = self._run_epoch(
                    train_loader, criterion, optimizer
                )

                val_loss, val_mae = float('nan'), float('nan')
                if val_loader is not None:
                    val_loss, val_mae = self._run_epoch(val_loader, criterion)

                scheduler.step()

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_mae'].append(train_mae)
                self.history['val_mae'].append(val_mae)
                self.history['lr'].append(lr_now)

                msg = (f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                       f"lr={lr_now:.2e} | "
                       f"train_loss={train_loss:.4f} mae={train_mae:.4f}")
                if val_loader is not None:
                    msg += f" | val_loss={val_loss:.4f} mae={val_mae:.4f}"
                print(msg, flush=True)

                ref_loss = val_loss if val_loader is not None else train_loss
                is_best = ref_loss + self.early_stopping_min_delta < best_val_loss
                if is_best:
                    best_val_loss = ref_loss
                    self.best_val_loss = ref_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if run_dir:
                    self._save_checkpoint(last_ckpt, epoch + 1, optimizer)
                    if is_best:
                        self._save_checkpoint(best_ckpt, epoch + 1, optimizer)

                if val_loader is not None and patience_counter >= self.early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(no val loss improvement for {patience_counter} epochs)."
                    )
                    break

            return {
                'history': self.history,
                'best_val_loss': self.best_val_loss,
                'epochs_trained': epoch + 1,
            }
        except KeyboardInterrupt:
            print("Training interrupted by user")
            if run_dir:
                self._save_checkpoint(last_ckpt, epoch + 1, optimizer)
            raise

    def _save_checkpoint(self, path, epoch, optimizer):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'norm_stats': self.norm_stats,
            'arch': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'horizon': self.horizon,
                'input_window': self.input_window,
                'dropout': self.dropout,
                'output_size': self.output_size,
                'num_lines': self.num_lines,
                'include_exits': self.include_exits,
                'add_time_feature': self.add_time_feature,
                'time_period_seconds': self.time_period_seconds,
                'bin_seconds': self.bin_seconds,
            },
        }
        torch.save(ckpt, path)
        if self.verbose:
            print(f"Saved checkpoint: {path}")

    def predict(self, series, steps=None, return_numpy=True,
                clip_negative=True, round_int=False):
        arr = np.asarray(series, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] == self.output_size and self.add_time_feature \
                and self.input_size == self.output_size + 2:
            arr = self._add_time_features(arr, self.bin_seconds)
        if arr.shape[1] != self.input_size:
            raise ValueError(
                f"Input feature size {arr.shape[1]} != model input_size {self.input_size}"
            )
        if arr.shape[0] < self.input_window:
            raise ValueError(
                f"Input series length {arr.shape[0]} < input_window {self.input_window}"
            )

        if self.norm_stats is not None:
            mean = np.asarray(self.norm_stats['mean'], dtype=np.float32)
            std = np.asarray(self.norm_stats['std'], dtype=np.float32)
        else:
            mean = np.zeros(arr.shape[1], dtype=np.float32)
            std = np.ones(arr.shape[1], dtype=np.float32)

        steps = int(steps) if steps is not None else self.horizon

        history_norm = (arr - mean) / std

        if isinstance(self.model, LinearPredictor):
            window = history_norm[-self.input_window:].reshape(1, self.input_window, -1)
            pred_norm = self.model.predict(window)[0]
            pred = pred_norm * std[:self.output_size] + mean[:self.output_size]
            if clip_negative:
                pred = np.maximum(pred, 0.0)
            if round_int:
                pred = np.round(pred)
            preds = [pred]
        elif isinstance(self.model, ARIMAPredictor):
            window = arr[-self.input_window:]
            preds = self.model.predict(
                window,
                steps=steps,
                clip_negative=clip_negative,
                round_int=round_int,
            )
            return preds if return_numpy else preds.tolist()
        else:
            self.model.eval()
            preds = []
            base_len = arr.shape[0]
            with torch.no_grad():
                while len(preds) < steps:
                    window = history_norm[-self.input_window:]
                    x = torch.from_numpy(window).unsqueeze(0).to(self.device)
                    out = self.model(x).cpu().numpy()[0]
                    for step_pred_norm in out:
                        out_mean = mean[:self.output_size]
                        out_std = std[:self.output_size]
                        step_pred = step_pred_norm * out_std + out_mean
                        if clip_negative:
                            step_pred = np.maximum(step_pred, 0.0)
                        if round_int:
                            step_pred = np.round(step_pred)
                        preds.append(step_pred.copy())

                        next_step_idx = base_len + len(preds) - 1
                        next_input = self._output_to_input(
                        step_pred, next_step_idx, self.bin_seconds
                    )
                    next_input_norm = (next_input - mean) / std
                    history_norm = np.vstack([history_norm,
                                              next_input_norm[np.newaxis, :]])
                    if len(preds) >= steps:
                        break

        preds = np.array(preds[:steps], dtype=np.float32)
        if self.output_size == 1:
            preds = preds.reshape(-1)
        return preds if return_numpy else preds.tolist()

    def generate(self, steps=300, seed_series=None, seed_length=None,
                 noise=0.0, clip_negative=True, round_int=True, seed=None):
        rng = np.random.default_rng(seed)

        if seed_series is None:
            length = seed_length or self.input_window
            flow_features = self.output_size
            seed_arr = np.zeros((length, flow_features), dtype=np.float32)
            if self.norm_stats is not None:
                base = np.asarray(self.norm_stats['mean'][:flow_features],
                                  dtype=np.float32)
                seed_arr += base
            seed_arr += rng.normal(0, max(noise, 0.1), size=seed_arr.shape).astype(np.float32)
            seed_arr = np.maximum(seed_arr, 0.0)
            seed_arr = self._add_time_features(seed_arr, self.bin_seconds) \
                if self.add_time_feature else seed_arr
        else:
            seed_arr = np.asarray(seed_series, dtype=np.float32)
            if seed_arr.ndim == 1:
                seed_arr = seed_arr.reshape(-1, 1)
            if seed_arr.shape[1] == self.output_size and self.add_time_feature:
                seed_arr = self._add_time_features(seed_arr, self.bin_seconds)

        forecast = self.predict(
            seed_arr,
            steps=steps,
            clip_negative=clip_negative,
            round_int=round_int,
        )

        if noise > 0:
            forecast = forecast + rng.normal(0, noise, size=forecast.shape).astype(np.float32)
            if clip_negative:
                forecast = np.maximum(forecast, 0.0)
            if round_int:
                forecast = np.round(forecast)

        return forecast

    @staticmethod
    def generate_synthetic_dataset(num_series=100, length=600, num_lines=4,
                                   bin_seconds=1.0, base_rate=0.3,
                                   period_seconds=120.0,
                                   amplitude=0.7, line_correlation=0.4,
                                   include_exits=False, add_time_feature=True,
                                   time_period_seconds=60.0,
                                   seed=None):
        rng = np.random.default_rng(seed)
        series_list = []
        meta_list = []

        for i in range(num_series):
            t = np.arange(length, dtype=np.float32) * bin_seconds

            line_phase_offsets = rng.uniform(0, 2 * math.pi, size=num_lines)
            line_base_rates = base_rate * rng.uniform(0.5, 1.5, size=num_lines)

            common_signal = amplitude * np.sin(
                2.0 * math.pi * t / period_seconds + rng.uniform(0, 2 * math.pi)
            ).astype(np.float32)

            entries = np.zeros((length, num_lines), dtype=np.float32)
            for line in range(num_lines):
                line_signal = amplitude * np.sin(
                    2.0 * math.pi * t / period_seconds + line_phase_offsets[line]
                ).astype(np.float32)
                mixed = line_correlation * common_signal + (1 - line_correlation) * line_signal
                rate = np.maximum(line_base_rates[line] * (1.0 + mixed), 0.0)
                entries[:, line] = rng.poisson(rate).astype(np.float32)

            if include_exits:
                exits = np.zeros_like(entries)
                for line in range(num_lines):
                    exit_rate = np.maximum(line_base_rates[line] * 0.9, 0.0)
                    exits[:, line] = rng.poisson(exit_rate * np.ones(length)).astype(np.float32)
                arr = np.concatenate([entries, exits], axis=1)
            else:
                arr = entries

            if add_time_feature:
                period = max(time_period_seconds, 1e-3)
                sin_t = np.sin(2.0 * math.pi * t / period).astype(np.float32)
                cos_t = np.cos(2.0 * math.pi * t / period).astype(np.float32)
                arr = np.concatenate([arr, np.stack([sin_t, cos_t], axis=1)], axis=1)

            series_list.append(arr)
            meta_list.append({
                'synthetic_id': i,
                'length': length,
                'bin_seconds': bin_seconds,
                'num_lines': num_lines,
                'base_rates': line_base_rates.tolist(),
            })

        return series_list, meta_list

    @staticmethod
    def synthetic_crossings_from_series(arr, num_lines=4, bin_seconds=1.0,
                                        synthetic_id=0, seed=None):
        rng = np.random.default_rng(seed)
        events = []
        next_vehicle_id = 1

        flow_arr = arr[:, :num_lines]
        for t_idx in range(flow_arr.shape[0]):
            t_seconds = t_idx * bin_seconds
            for line in range(num_lines):
                count = int(round(flow_arr[t_idx, line]))
                for _ in range(count):
                    sub_offset = rng.uniform(0.0, bin_seconds)
                    direction = rng.choice(['from_left', 'from_right'])
                    events.append({
                        'time': float(t_seconds + sub_offset),
                        'frame': int((t_seconds + sub_offset) * 30),
                        'line_no': line,
                        'vehicle_id': next_vehicle_id,
                        'class_name': rng.choice(['car', 'truck', 'bus'],
                                                  p=[0.85, 0.1, 0.05]),
                        'direction': direction,
                        'video_id': f'synthetic_{synthetic_id}',
                    })
                    next_vehicle_id += 1

        events.sort(key=lambda e: (e['time'], e['line_no']))
        return events

    def predict_for_road(self, flow_result, road_idx, steps=None,
                         series_key='time_series_entries'):
        series = flow_result['roads'][road_idx][series_key]
        return self.predict(series, steps=steps)

    def evaluate(self, series_list, batch_size=32):
        ds = TrafficFlowDataset(
            series_list=series_list,
            input_window=self.input_window,
            horizon=self.horizon,
            stride=1,
            normalize=True,
            stats=self.norm_stats,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        criterion = nn.MSELoss()
        loss, mae = self._run_epoch(loader, criterion)

        std = np.asarray(self.norm_stats['std'] if self.norm_stats else [1.0],
                         dtype=np.float32).mean()
        return {
            'mse_normalized': loss,
            'mae_normalized': mae,
            'mae_original_scale': mae * float(std),
            'rmse_original_scale': float(np.sqrt(loss)) * float(std),
            'num_samples': len(ds),
        }

    def save(self, checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
        if isinstance(self.model, (LinearPredictor, ARIMAPredictor)):
            torch.save(self.model, checkpoint_path)
            if self.verbose:
                print(f"Model saved to {checkpoint_path}")
            return

        ckpt = {
            'epoch': self.start_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'norm_stats': self.norm_stats,
            'arch': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'horizon': self.horizon,
                'input_window': self.input_window,
                'dropout': self.dropout,
                'output_size': self.output_size,
                'num_lines': self.num_lines,
                'include_exits': self.include_exits,
                'add_time_feature': self.add_time_feature,
                'time_period_seconds': self.time_period_seconds,
                'bin_seconds': self.bin_seconds,
            },
        }
        torch.save(ckpt, checkpoint_path)
        if self.verbose:
            print(f"Model saved to {checkpoint_path}")

    def _safe_load_checkpoint(self, checkpoint_path):
        try:
            return torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception:
            import torch.serialization
            with torch.serialization.safe_globals([ARIMAPredictor, LinearPredictor]):
                return torch.load(checkpoint_path, map_location=self.device, weights_only=False)

    def load(self, checkpoint_path):
        ckpt = self._safe_load_checkpoint(checkpoint_path)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            cfg = ckpt.get('arch', {})
            self.input_size = cfg.get('input_size', self.input_size)
            self.hidden_size = cfg.get('hidden_size', self.hidden_size)
            self.num_layers = cfg.get('num_layers', self.num_layers)
            self.horizon = cfg.get('horizon', self.horizon)
            self.input_window = cfg.get('input_window', self.input_window)
            self.dropout = cfg.get('dropout', self.dropout)
            self.output_size = cfg.get('output_size', self.output_size)
            self.num_lines = cfg.get('num_lines', self.num_lines)
            self.include_exits = cfg.get('include_exits', self.include_exits)
            self.add_time_feature = cfg.get('add_time_feature', self.add_time_feature)
            self.time_period_seconds = cfg.get('time_period_seconds', self.time_period_seconds)
            self.bin_seconds = cfg.get('bin_seconds', self.bin_seconds)
            self.model = TrafficLSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                horizon=self.horizon,
                dropout=self.dropout,
                output_size=self.output_size,
            ).to(self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.history = ckpt.get('history', self.history)
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            self.start_epoch = ckpt.get('epoch', 0)
            self.norm_stats = ckpt.get('norm_stats', None)
        elif isinstance(ckpt, (LinearPredictor, ARIMAPredictor)):
            self.model = ckpt
        else:
            self.model.load_state_dict(ckpt)
        if self.verbose:
            print(f"Model loaded from {checkpoint_path}")

    def save_training_metrics(self, run_dir):
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, 'history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        return [path]

    def create_run_summary(self, run_dir, model_name='TrafficPredictor'):
        os.makedirs(run_dir, exist_ok=True)
        summary_path = os.path.join(run_dir, 'run_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("TRAFFIC FLOW PREDICTION RUN SUMMARY\n")
            f.write(f"Run Directory: {run_dir}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Num lines: {self.num_lines}\n")
            f.write(f"Include exits: {self.include_exits}\n")
            f.write(f"Add time feature: {self.add_time_feature}\n")
            f.write(f"Time period (s): {self.time_period_seconds}\n")
            f.write(f"Bin seconds: {self.bin_seconds}\n")
            f.write(f"Input window: {self.input_window}\n")
            f.write(f"Horizon: {self.horizon}\n")
            f.write(f"Input size: {self.input_size}\n")
            f.write(f"Output size: {self.output_size}\n")
            f.write(f"Hidden size: {self.hidden_size}\n")
            f.write(f"Layers: {self.num_layers}\n")
            f.write(f"Dropout: {self.dropout}\n")
            f.write(f"Batch size: {self.config.batch_size}\n")
            f.write(f"Epochs: {self.config.num_epochs}\n")
            f.write(f"Learning rate: {self.config.lr}\n")
            f.write(f"Weight decay: {self.config.weight_decay}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Best val loss: {self.best_val_loss:.6f}\n")
            if self.norm_stats:
                f.write(f"Norm mean: {self.norm_stats['mean']}\n")
                f.write(f"Norm std:  {self.norm_stats['std']}\n")
        return summary_path