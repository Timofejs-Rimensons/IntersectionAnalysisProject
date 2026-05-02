# Quick Reference - YOLOv8 Training Commands

## Essential Commands

```bash
# 1️⃣  View help and examples
python model_forgery.py

# 2️⃣  Train new model (creates timestamped folder in weights/)
python model_forgery.py --mode train --verbose

# 3️⃣  Resume interrupted training
python model_forgery.py --mode train --resume weights/YYYY-MM-DD_HHMMSS_track/

# 4️⃣  Run inference on video
python model_forgery.py --mode inference --source video.mp4 --weights best.pt

# 5️⃣  Convert VisDrone dataset (one-time)
python model_forgery.py --mode convert \
  --seq-dir datasets/VisDrone2019-MOT-train/sequences \
  --ann-dir datasets/VisDrone2019-MOT-train/annotations

# 6️⃣  List all training runs
ls -lh weights/ | grep track

# 7️⃣  View latest training results
cat weights/$(ls -t weights/ | grep track | head -1)/results.csv

# 8️⃣  View training plot
open weights/$(ls -t weights/ | grep track | head -1)/results.png  # MacOS
xdg-open weights/$(ls -t weights/ | grep track | head -1)/results.png  # Linux
```

## Output Directory Structure

```
weights/
└── 2024-01-15_143022_track/
    ├── best.pt                    ← Use this for inference
    ├── last.pt
    ├── results.csv                ← Training metrics
    ├── results.png                ← Training curves plot
    ├── confusion_matrix.png
    ├── run_summary.txt            ← Configuration & metadata
    └── examples_predictions/
```

## File Locations

| What | Where |
|------|-------|
| Best weights | `weights/LATEST_track/best.pt` |
| Training metrics | `weights/LATEST_track/results.csv` |
| Training plot | `weights/LATEST_track/results.png` |
| Config summary | `weights/LATEST_track/run_summary.txt` |
| Sample predictions | `weights/LATEST_track/examples_predictions/` |
| Inference results | `weights/inference_*/**` |

## Key Features

[+] **Timestamped directories** - Each training run is separate (no overwriting)  
[+] **All metrics saved** - best.pt, last.pt, results.csv, results.png  
[+] **Easy resume** - Continue from checkpoint with `--resume`  
[+] **Clean structure** - Everything in `weights/` folder  
[+] **Automatic tracking** - History preserved in run_summary.txt  

---

For detailed guide, see: `TRAINING_GUIDE.md`
