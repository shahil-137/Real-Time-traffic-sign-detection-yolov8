import json
import os
import shutil
import cv2
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

ANNOT_PATH = os.path.join(PROJECT_DIR, "dataset", "tt100k_2021", "annotations_all.json")
TT100K_DIR = os.path.join(PROJECT_DIR, "dataset", "tt100k_2021")
YOLO_DIR = os.path.join(PROJECT_DIR, "dataset", "yolo")

IMG_TRAIN = os.path.join(YOLO_DIR, "images", "train")
IMG_VAL   = os.path.join(YOLO_DIR, "images", "val")
LBL_TRAIN = os.path.join(YOLO_DIR, "labels", "train")
LBL_VAL   = os.path.join(YOLO_DIR, "labels", "val")

MIN_INSTANCES = 80
MAX_CLASSES = 25 


def to_float(v):
    try:
        return float(v)
    except:
        return 0.0


# ---------- LOAD ANNOTATIONS ----------
with open(ANNOT_PATH, "r") as f:
    data = json.load(f)

# ---------- COUNT INSTANCES ----------
counts = defaultdict(int)
for img in data["imgs"].values():
    for obj in img.get("objects", []):
        counts[obj.get("category")] += 1

# ---------- BUILD CLASS MAP ----------
CLASS_MAP = {}
idx = 0

for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
    if cnt < MIN_INSTANCES:
        continue
    if idx >= MAX_CLASSES:
        break
    CLASS_MAP[cls] = idx
    idx += 1


print("\nSelected classes (>=50 instances):")
for c, i in CLASS_MAP.items():
    print(f"{i:2d} -> {c} ({counts[c]})")

print("\nTotal classes:", len(CLASS_MAP))

# ---------- CREATE DIRS ----------
for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    os.makedirs(d, exist_ok=True)

# ---------- CONVERT ----------
for img_info in data["imgs"].values():
    path = img_info.get("path", "")
    if not (path.startswith("train") or path.startswith("test")):
        continue

    split = "train" if path.startswith("train") else "val"

    img_src = os.path.join(TT100K_DIR, path)
    if not os.path.exists(img_src):
        continue

    img_dst = IMG_TRAIN if split == "train" else IMG_VAL
    lbl_dst = LBL_TRAIN if split == "train" else LBL_VAL

    shutil.copy(img_src, img_dst)

    image = cv2.imread(img_src)
    if image is None:
        continue

    H, W = image.shape[:2]
    label_lines = []

    for obj in img_info.get("objects", []):
        cls = obj.get("category")
        if cls not in CLASS_MAP:
            continue

        bbox = obj.get("bbox", {})
        xmin = to_float(bbox.get("xmin"))
        ymin = to_float(bbox.get("ymin"))
        xmax = to_float(bbox.get("xmax"))
        ymax = to_float(bbox.get("ymax"))

        if xmax <= xmin or ymax <= ymin:
            continue

        xc = ((xmin + xmax) / 2) / W
        yc = ((ymin + ymax) / 2) / H
        bw = (xmax - xmin) / W
        bh = (ymax - ymin) / H

        label_lines.append(
            f"{CLASS_MAP[cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        )

    if label_lines:
        name = os.path.splitext(os.path.basename(path))[0] + ".txt"
        with open(os.path.join(lbl_dst, name), "w") as f:
            f.write("\n".join(label_lines))

# ---------- WRITE data.yaml ----------
yaml_path = os.path.join(YOLO_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {YOLO_DIR}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write(f"nc: {len(CLASS_MAP)}\n\n")
    f.write("names:\n")
    for cls, idx in sorted(CLASS_MAP.items(), key=lambda x: x[1]):
        f.write(f"  {idx}: {cls}\n")

print("\n✅ YOLO dataset conversion completed.")
