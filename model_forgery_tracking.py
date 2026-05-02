
"""
YOLOv8-based Multi-Object Tracking for VisDrone Footage

This script demonstrates multi-object tracking using YOLOv8 on aerial imagery
from the VisDrone MOT dataset.

Usage:
    # Training for tracking
    python model_forgery.py --mode train --data datasets/visdrone_yolo/data.yaml
    
    # Inference
    python model_forgery.py --mode inference --source /path/to/video.mp4 --weights best.pt
    
    # First, convert VisDrone MOT dataset to YOLO format (recommended)
    python model_forgery.py --visdrone-convert --seq-dir datasets/VisDrone2019-MOT-train/sequences --ann-dir datasets/VisDrone2019-MOT-train/annotations
"""

import os
import sys

# Force unbuffered output for real-time training feedback
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import argparse
import json
import yaml
from pathlib import Path
import shutil

from tqdm import tqdm
from services.TrackingModel import TrackingModel
from services.VisDroneMOTCocoConvertor import VisDroneMOTCocoConvertor
from config.Config import Config


def create_yolo_data_yaml(config, output_path):
    """
    Create YOLO data.yaml file from COCO annotations.
    
    Args:
        config: Config object with paths
        output_path: Path to save data.yaml
    """
    with open(os.path.join(config.data_dir, config.train_json), 'r') as f:
        coco = json.load(f)
    
    class_names = {cat['id']: cat['name'] for cat in coco['categories']}
    num_classes = len(class_names)
    
    data_yaml = {
        'path': str(Path(config.data_dir).absolute()),
        'train': f'images/train',
        'val': f'images/val',
        'nc': num_classes,
        'names': class_names
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created data.yaml: {output_path}")
    print(f"Classes ({num_classes}): {list(class_names.values())}")
    
    return output_path





def train_tracking(config, data_yaml, resume_from=None, verbose=False):
    """
    Train YOLOv8 for multi-object tracking.
    
    YOLOv8 tracking uses the detection backbone and adds a tracker.
    The tracker (ByteTrack or BoTSORT) maintains object identities across frames.
    
    All outputs (weights, metrics, visualizations) are saved to:
    weights/<YYYY-MM-DD>_<HHMMSS>_track/
    
    Args:
        config: Config object
        data_yaml: Path to data.yaml file
        resume_from: Resume training from existing run directory (optional)
        verbose: Enable verbose output
        
    Returns:
        Tuple of (model, run_directory)
    """
    print("\n" + "="*70)
    print("TRAINING YOLOV8 FOR MULTI-OBJECT TRACKING")
    print("="*70)
    print("\nNote: YOLOv8 tracking extends detection with temporal consistency.")
    print("Tracker options: ByteTrack (fast, recommended), BoTSORT (accurate)")
    
    # Get or create run directory
    run_dir = config.get_run_directory(task='track', resume_from=resume_from)
    print(f"\nTraining outputs will be saved to: {run_dir}\n", flush=True)
    
    print("[*] Initializing TrackingModel...", flush=True)
    model = TrackingModel(config, task='track', verbose=True)
    
    print("[*] Starting training loop (this may take a few minutes)...\n", flush=True)
    try:
        results = model.train(
            train_data_yaml=data_yaml,
            resume_from_last=False,  # We handle resume with run_dir
            run_dir=run_dir,
            class_weights_labels_dir="datasets/visdrone_yolo/labels/train"  # Disable for faster startup; YOLO handles class balance
        )
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user", flush=True)
        raise
    except Exception as e:
        print(f"\n\n[!] Training failed: {e}", flush=True)
        raise
    
    # Save best model in weights folder
    best_model_path = os.path.join(run_dir, 'best.pt')
    last_model_path = os.path.join(run_dir, 'last.pt')
    
    # YOLO already saves these in run_dir/weights/ directory
    # Create symbolic links or copy to run_dir root for easier access
    yolo_weights_dir = os.path.join(run_dir, 'weights')
    if os.path.exists(yolo_weights_dir):
        yolo_best = os.path.join(yolo_weights_dir, 'best.pt')
        yolo_last = os.path.join(yolo_weights_dir, 'last.pt')
        
        if os.path.exists(yolo_best):
            import shutil
            shutil.copy2(yolo_best, best_model_path)
            print(f"\nBest model: {best_model_path}")
        
        if os.path.exists(yolo_last):
            import shutil
            shutil.copy2(yolo_last, last_model_path)
            print(f"Last model: {last_model_path}")
    
    # Save training metrics and create summary
    model.save_training_metrics(run_dir, results)
    model.create_run_summary(run_dir, data_yaml=data_yaml, model_name='yolov8m')
    
    print(f"\n{'='*70}")
    print(f"Training completed successfully!")
    print(f"Run directory: {run_dir}")
    print(f"{'='*70}")
    
    return model, run_dir





def inference_track(config, source, weights='best.pt', run_dir=None, verbose=False):
    """
    Run tracking inference on video.
    
    Args:
        config: Config object
        source: Video path
        weights: Model weights filename (best.pt or last.pt)
        run_dir: Directory containing the model (for loading best.pt/last.pt)
                If None, looks in most recent weights folder
        verbose: Enable verbose output
    """
    print("\n" + "="*70)
    print("MULTI-OBJECT TRACKING INFERENCE")
    print("="*70)
    
    # Determine model path
    if run_dir:
        model_path = os.path.join(run_dir, weights)
    else:
        # Find most recent run directory
        weights_dir = 'weights'
        if os.path.exists(weights_dir):
            runs = sorted([d for d in os.listdir(weights_dir) 
                          if os.path.isdir(os.path.join(weights_dir, d)) and '_track' in d],
                         reverse=True)
            if runs:
                run_dir = os.path.join(weights_dir, runs[0])
                model_path = os.path.join(run_dir, weights)
            else:
                raise ValueError("No tracking runs found in weights/ directory")
        else:
            raise ValueError("weights/ directory not found")
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model not found: {model_path}")
    
    print(f"Using model: {model_path}\n")
    
    model = TrackingModel(config, checkpoint_path=model_path, task='track', verbose=verbose)
    
    results = model.track(source, conf=0.5, iou=0.45, tracker='bytetrack.yaml')
    
    # Create inference output directory
    output_base = os.path.dirname(run_dir) if run_dir else 'weights/inference'
    output_dir = os.path.join(output_base, f'inference_{os.path.basename(source).split(".")[0]}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Export results in multiple formats
    mot_txt = os.path.join(output_dir, 'tracking_results.txt')
    model.export_tracks_to_mot(results, mot_txt, video_name=Path(source).stem)
    
    coco_track_json = os.path.join(output_dir, 'tracking_results.json')
    model.export_tracks_to_coco_tracking(
        results, coco_track_json, source, Path(source).stem
    )
    
    # Visualize
    output_video = os.path.join(output_dir, 'tracking_output.mp4')
    model.visualize_tracks(source, results, output_video)
    
    # Print summary
    all_track_ids = set()
    for result in results:
        if result.boxes.id is not None:
            all_track_ids.update(int(tid[0]) for tid in result.boxes.id)
    
    print(f"\nTracking Results Summary:")
    print(f"  Frames processed: {len(results)}")
    print(f"  Unique track IDs: {len(all_track_ids)}")
    total_detections = sum(len(r.boxes) for r in results)
    print(f"  Total detections: {total_detections}")
    print(f"\nResults saved to: {output_dir}")
    
    return results, output_dir


def convert_visdrone_mot_to_yolo_format(
    seq_dir,
    ann_dir,
    output_dir,
    val_split=0.05 # Keep val_split for visdrone_mot_to_coco_train_val
):
    """
    Convert VisDrone MOT dataset to YOLO format.

    YOLO format expects:
        images/
            train/
                img1.jpg
                ...
            val/
                img2.jpg
                ...
        labels/
            train/
                img1.txt (YOLO format: <class_id> <x_center> <y_center> <width> <height> normalized)
                ...
            val/
                img2.txt
                ...

    Args:
        seq_dir: Path to sequence images
        ann_dir: Path to annotations
        output_dir: Output directory for YOLO format
        val_split: Validation split ratio for video sequences
    """
    print("\n" + "="*70)
    print("CONVERTING VISDRONE MOT TO YOLO FORMAT")
    print("="*70)

    convertor = VisDroneMOTCocoConvertor()

    # First convert to COCO Tracking intermediate format with train/val split
    # This creates output_dir/annotations/tracking_train.json and tracking_val.json
    convertor.visdrone_mot_to_coco_train_val(
        seq_dir=seq_dir,
        ann_dir=ann_dir,
        output_dir=output_dir,
        val_split=val_split
    )

    # Convert COCO train JSON to YOLO format
    train_coco_json = os.path.join(output_dir, 'coco_annotations', 'tracking_train.json')
    convert_coco_to_yolo_format(
        coco_json=train_coco_json,
        output_dir=output_dir,
        img_subdir='images/train',
        label_subdir='labels/train',
        seq_dir=seq_dir
    )

    # Convert COCO val JSON to YOLO format
    val_coco_json = os.path.join(output_dir, 'coco_annotations', 'tracking_val.json')
    convert_coco_to_yolo_format(
        coco_json=val_coco_json,
        output_dir=output_dir,
        img_subdir='images/val',
        label_subdir='labels/val',
        seq_dir=seq_dir
    )
    print(f"VisDrone MOT converted to YOLO format in: {output_dir}")


def convert_coco_to_yolo_format(coco_json, output_dir, img_subdir, label_subdir, seq_dir):
    """
    Convert COCO JSON annotations to YOLO txt format for a single split.
    Also copies the corresponding image files.

    CRITICAL: VisDrone sequences all share the same frame filenames
    (0000001.jpg, 0000002.jpg, ...). We must prefix with the sequence
    name to avoid overwriting.
    """
    with open(coco_json, 'r') as f:
        coco = json.load(f)

    target_img_dir = os.path.join(output_dir, img_subdir)
    target_lbl_dir = os.path.join(output_dir, label_subdir)

    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_lbl_dir, exist_ok=True)

    # Build annotations lookup
    anns_by_img = {}
    for ann in coco['annotations']:
        iid = ann['image_id']
        if iid not in anns_by_img:
            anns_by_img[iid] = []
        anns_by_img[iid].append(ann)

    copied = 0
    skipped = 0
    class_counts = {}

    for img in tqdm(coco['images'], desc=f"Converting {img_subdir.split('/')[-1]} COCO→YOLO"):
        img_id = img['id']
        img_w, img_h = img['width'], img['height']
        anns = anns_by_img.get(img_id, [])

        # ── FIX: Create unique filename by joining sequence + frame ──
        # img['file_name'] = "uav0000013_00000_v/0000001.jpg"
        # unique_name      = "uav0000013_00000_v_0000001.jpg"
        rel_path = img['file_name']                          # e.g. "uav0000013_00000_v/0000001.jpg"
        parts = Path(rel_path).parts                         # ('uav0000013_00000_v', '0000001.jpg')
        unique_name = "_".join(parts)                        # "uav0000013_00000_v_0000001.jpg"
        unique_stem = Path(unique_name).stem                 # "uav0000013_00000_v_0000001"

        # Source and target paths
        original_img_path = os.path.join(seq_dir, rel_path)
        target_img_path = os.path.join(target_img_dir, unique_name)
        label_path = os.path.join(target_lbl_dir, f"{unique_stem}.txt")

        # Copy image
        if not os.path.exists(original_img_path):
            print(f"  Warning: not found: {original_img_path}")
            skipped += 1
            continue

        shutil.copyfile(original_img_path, target_img_path)
        copied += 1

        # Write YOLO label
        with open(label_path, 'w') as lbl_file:
            for ann in anns:
                x, y, w, h = ann['bbox']

                center_x = (x + w / 2.0) / img_w
                center_y = (y + h / 2.0) / img_h
                width = w / img_w
                height = h / img_h

                # Clamp to [0, 1]
                center_x = max(0.0, min(1.0, center_x))
                center_y = max(0.0, min(1.0, center_y))
                width    = max(0.0, min(1.0, width))
                height   = max(0.0, min(1.0, height))

                class_id = int(ann['category_id'])
                lbl_file.write(
                    f"{class_id} {center_x:.6f} {center_y:.6f} "
                    f"{width:.6f} {height:.6f}\n"
                )

                # Track class distribution
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

    # ── Summary ───────────────────────────────────────────
    print(f"\n  Images copied:  {copied:,}")
    print(f"  Images skipped: {skipped:,}")
    print(f"  Labels created: {copied:,}")
    print(f"\n  Per-class annotation counts in YOLO labels:")

    # Get category names from COCO
    cat_names = {c['id']: c['name'] for c in coco.get('categories', [])}
    for cid in sorted(class_counts.keys()):
        name = cat_names.get(cid, f"class_{cid}")
        print(f"    {cid}: {name:<20s} {class_counts[cid]:>8,}")

    total = sum(class_counts.values())
    print(f"    {'TOTAL':<23s} {total:>8,}")

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Multi-Object Tracking')
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'convert'], default='train',
                       help='Mode: train, inference, or convert')
    parser.add_argument('--data', type=str, default='datasets/visdrone_yolo/data.yaml',
                       help='Path to data.yaml for training')
    parser.add_argument('--source', type=str,
                       help='Video path for inference')
    parser.add_argument('--weights', type=str, default='best.pt',
                       help='Model weights filename (best.pt or last.pt)')
    parser.add_argument('--run-dir', type=str,
                       help='Training run directory (for resume or inference)')
    parser.add_argument('--resume', type=str,
                       help='Resume training from existing run directory')
    parser.add_argument('--visdrone-convert', action='store_true',
                       help='Convert VisDrone MOT to YOLO format first')
    parser.add_argument('--seq-dir', type=str,
                       help='Sequence images directory (for UAV conversion)')
    parser.add_argument('--ann-dir', type=str,
                       help='Annotations directory (for UAV conversion)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Config
    config = Config()
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Convert VisDrone dataset if in convert mode
    if args.mode == 'convert':
        if not args.seq_dir or not args.ann_dir:
            raise ValueError("--seq-dir and --ann-dir required for convert mode")
        convert_visdrone_mot_to_yolo_format(
            seq_dir=args.seq_dir,
            ann_dir=args.ann_dir,
            output_dir='datasets/visdrone_yolo'
        )
        print("[*] Dataset conversion complete. You can now train with:")
        print("    python model_forgery.py --mode train --verbose")
        return
    
    # For train/inference modes, ensure data.yaml exists
    # Create data.yaml if needed
    if not os.path.exists(args.data):
        args.data = create_yolo_data_yaml(config, args.data)
    
    # Training
    if args.mode == 'train':
        model, run_dir = train_tracking(config, args.data, resume_from=args.resume, verbose=True)
    
    # Inference
    elif args.mode == 'inference':
        if not args.source:
            raise ValueError("--source required for inference mode")
        
        results, output_dir = inference_track(config, args.source, args.weights, args.run_dir, verbose=args.verbose)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MULTI-OBJECT TRACKING - VISDRONE MOT")
    print("="*70)
    
    # Example usage if no arguments provided:
    import sys
    if len(sys.argv) == 1:
        print("\n" + "="*70)
        print("USAGE GUIDE")
        print("="*70)
        
        print("\n1. CONVERT VISDRONE DATASET TO YOLO FORMAT:")
        print("   python model_forgery.py --mode convert \\")
        print("     --seq-dir datasets/VisDrone2019-MOT-train/sequences \\")
        print("     --ann-dir datasets/VisDrone2019-MOT-train/annotations")
        
        print("\n2. TRAIN TRACKING MODEL (NEW RUN):")
        print("   python model_forgery.py --mode train --verbose")
        print("   ")
        print("   Output structure:")
        print("   weights/")
        print("   └── 2024-01-15_143022_track/        # New training run (timestamp)")
        print("       ├── best.pt                      # Best model weights")
        print("       ├── last.pt                      # Last model weights")
        print("       ├── weights/")
        print("       │   ├── best.pt")
        print("       │   └── last.pt")
        print("       ├── results.csv                  # Training metrics")
        print("       ├── results.png                  # Training curves plot")
        print("       ├── confusion_matrix.png         # Confusion matrix")
        print("       ├── run_summary.txt              # Run summary")
        print("       └── examples_predictions/        # Sample predictions")
        
        print("\n3. RESUME TRAINING FROM EXISTING RUN:")
        print("   python model_forgery.py --mode train \\")
        print("     --resume weights/2024-01-15_143022_track/")
        
        print("\n4. RUN INFERENCE ON VIDEO:")
        print("   python model_forgery.py --mode inference \\")
        print("     --source path/to/video.mp4 \\")
        print("     --weights best.pt \\")
        print("     --run-dir weights/2024-01-15_143022_track/")
        print("   ")
        print("   (If --run-dir not specified, uses most recent track run)")
        
        print("\n5. LIST ALL TRAINING RUNS:")
        print("   ls -lh weights/")
        
        print("\n" + "="*70)
        print("KEY FEATURES:")
        print("="*70)
        print("[+] All weights saved in weights/ folder")
        print("[+] Each training run in separate timestamped directory")
        print("[+] Metrics automatically saved (results.png, results.csv)")
        print("[+] Model summary and config preserved in run_summary.txt")
        print("[+] Easy to resume training from checkpoints")
        print("[+] Inference results saved in separate inference/ folder")
        
        print("\n" + "="*70 + "\n")
        sys.exit(0)
    
    main()