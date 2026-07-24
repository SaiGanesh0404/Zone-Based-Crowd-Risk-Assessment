# 2) SAVE THIS AS: make_metrics.py   (put it inside your project folder)
# Then run:  python make_metrics.py

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ----------------- EDIT ONLY THIS (your project folder has these) -----------------
IMG_DIR = r"datasets\coco2017\images\val2017"   # images
LBL_DIR = r"datasets\coco2017\labels\val2017"   # yolo txt labels

BEST_PT = r"runs\detect\train\weights\best.pt"
MODEL_PATH = BEST_PT if os.path.exists(BEST_PT) else r"models\yolov8m.pt"

OUT_DIR = r"outputs\metrics"
THRESH = 0.50

MAX_IMAGES = None   # e.g. 2000 to test faster
IMGSZ = 640
# -------------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
if MAX_IMAGES is not None:
    img_paths = img_paths[:MAX_IMAGES]
assert len(img_paths) > 0, f"No images found in: {IMG_DIR}"

def has_person(txt_path: str) -> int:
    if not os.path.exists(txt_path):
        return 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls = int(float(line.split()[0]))
            if cls == 0:   # COCO class 0 = person
                return 1
    return 0

# Ground truth
y_true = []
for img in tqdm(img_paths, desc="Reading labels"):
    stem = os.path.splitext(os.path.basename(img))[0]
    txtp = os.path.join(LBL_DIR, stem + ".txt")
    y_true.append(has_person(txtp))
y_true = np.array(y_true, dtype=int)

# Model (Ultralytics will use GPU automatically if available)
model = YOLO(MODEL_PATH)

scores = []
for img in tqdm(img_paths, desc="YOLO inference"):
    r = model.predict(img, imgsz=IMGSZ, conf=0.001, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        scores.append(0.0)
        continue
    cls = r.boxes.cls.detach().cpu().numpy().astype(int)
    conf = r.boxes.conf.detach().cpu().numpy().astype(float)
    pc = conf[cls == 0]
    scores.append(float(pc.max()) if pc.size else 0.0)

scores = np.array(scores, dtype=float)

# Confusion Matrix
y_pred = (scores >= THRESH).astype(int)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=["No Person", "Person"])
plt.figure(figsize=(6, 6))
disp.plot(values_format="d", cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix (thr={THRESH})")
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=200)
plt.show()

# ROC
fpr, tpr, _ = roc_curve(y_true, scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Person Present)")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(OUT_DIR, "roc.png")
plt.savefig(roc_path, dpi=200)
plt.show()

# Loss vs Metrics (if trained)
results_csv = r"runs\detect\train\results.csv"
lvm_path = os.path.join(OUT_DIR, "loss_vs_metric.png")

if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    metric_cols = [c for c in df.columns if any(k in c.lower() for k in ["precision", "recall", "map"])]

    plt.figure(figsize=(9, 5))
    for c in loss_cols:
        plt.plot(df[c].to_numpy(), label=c)
    for c in metric_cols:
        plt.plot(df[c].to_numpy(), label=c)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Loss vs Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(lvm_path, dpi=200)
    plt.show()

print("Saved:")
print(cm_path)
print(roc_path)
print(lvm_path if os.path.exists(results_csv) else "(loss_vs_metric skipped: results.csv not found)")