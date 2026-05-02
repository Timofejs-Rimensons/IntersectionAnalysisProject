import torch
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    """
    Configuration for RetinaNet object detection model training and inference.
    
    Data Paths:
        data_dir: Root directory containing dataset
        train_json: COCO annotation JSON for training set
        val_json: COCO annotation JSON for validation set
        output_dir: Directory to save checkpoints and results
    
    Model Architecture:
        trainable_backbone_layers: Number of backbone layers to fine-tune (0=frozen)
        num_classes: Number of object classes (5 for traffic: person, bike, car, truck, bus)
    
    Training Hyperparameters:
        batch_size: Batch size per GPU
        num_epochs: Total training epochs
        lr: Learning rate for AdamW optimizer
        weight_decay: L2 regularization
        grad_clip: Gradient clipping max norm
        accumulation_steps: Gradient accumulation for larger effective batch
    
    Validation:
        val_interval: Validate every N epochs
        epoch_size: Samples per training epoch
        val_epoch_size: Samples per validation epoch
    
    Data Processing:
        img_size: Input image size (H, W)
        use_augmentation: Enable albumentations augmentation pipeline
        aug_prob: Probability of applying augmentations
        num_workers: DataLoader workers
        mixed_precision: Enable AMP for faster training
    
    Inference:
        device: cuda or cpu (auto-detected)
        class_names: List of class names for all classes
        colors: RGB hex colors for visualization
    """
    # Data paths
    data_dir: str = "datasets/visdrone_yolo"
    train_json: str = "coco_annotations/tracking_train.json"
    val_json: str = "coco_annotations/tracking_val.json"
    train_img_dir: str = "images"
    val_img_dir: str = "images"
    output_dir: str = "weights"

    # Model architecture
    model: str = "yolo26s"
    trainable_backbone_layers: int = 0

    # Training hyperparameters
    batch_size: int = 8  # Increased from 1 for better GPU utilization
    num_epochs: int = 30000
    epoch_size: int = 1000
    val_epoch_size: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 0.1
    num_workers: int = 8  # Increased from 1 for parallel data loading
    val_interval: int = 2
    accumulation_steps: int = 4
    mixed_precision: bool = True

    # Data processing
    img_size: Tuple[int, int] = (640, 640)  # Reduced from (1280, 1280) for faster augmentation
    max_video_dim: int = 1280  # Cap video resolution to 720p / 1280px long side for tracking
    track_gap_interpolation_frames: int = 3  # Fill short track occlusions up to this many frames
    use_augmentation: bool = True
    aug_prob: float = 0.5
    mosaic_prob: float = 0.0
    
    pred_input_window: int = 300   # past timesteps fed to the predictor
    pred_horizon: int = 10        # how many steps to predict at once
    pred_input_size: int = 1      # features per timestep (1 for univariate flow)
    pred_hidden_size: int = 64
    pred_num_layers: int = 2
    pred_dropout: float = 0.2
    pred_output_size: int = 1
    pred_add_time_feature: bool = True
    pred_time_period_seconds: float = 60.0
    pred_bin_seconds: float = 1.0
    pred_augment_time_shift: bool = True
    pred_time_shift_max_phase: float = 1.0  # maximum random phase shift in cycles

    # Device and visualization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    class_names: List[str] = field(default_factory=lambda: [
        "person", "bicycle", "car", "truck", "bus"
    ])

    colors: List[str] = field(default_factory=lambda: [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231'
    ])

    num_classes: int = 5

    def get_run_directory(self, task='track', resume_from=None):
        """
        Get or create a timestamped directory for this training run.
        All weights, metrics, and visualizations will be saved here.
        
        Args:
            task: 'track' or 'detect'
            resume_from: If provided, resume training in this directory instead of creating new
            
        Returns:
            Path to the run directory (e.g., weights/2024-01-15_143022_track/)
        """
        if resume_from:
            # Resume existing training
            if not os.path.exists(resume_from):
                raise ValueError(f"Resume directory does not exist: {resume_from}")
            return resume_from
        
        # Create new timestamped directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = os.path.join("weights", f"{timestamp}_{task}")
        os.makedirs(run_dir, exist_ok=True)
        
        return run_dir

    # ============================================================================
    # FLOW PREDICTION CONFIGURATION
    # ============================================================================

    # Model selection
    pred_model_type: str = "arima"  # "arima", "linear" or "lstm"

    # ARIMA hyperparameters (if pred_model_type == "arima")
    pred_arima_order: Tuple[int, int, int] = (1, 1, 0)
    pred_arima_seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
    pred_arima_trend: str = "c"

    # LSTM hyperparameters (if pred_model_type == "lstm")
    pred_hidden_size: int = 64
    pred_num_layers: int = 2
    pred_dropout: float = 0.2
    pred_learning_rate: float = 1e-3
    pred_batch_size: int = 16
    pred_epochs: int = 100
    pred_early_stopping_patience: int = 10

    # Linear model hyperparameters (if pred_model_type == "linear")
    pred_lookback: int = 5  # Number of past time steps to use as features
    pred_forecast_horizon: int = 1  # Predict next N steps

    # Data processing
    pred_bin_seconds: float = 1.0  # Time bin size in seconds
    pred_augment_time_shift: bool = True  # Enable time shift augmentation
    pred_augment_max_shift: int = 3  # Max bins to shift for augmentation

    # Training
    pred_train_split: float = 0.7  # Train/val split ratio
    pred_split_by_intersection: bool = True  # Split by video to prevent leakage