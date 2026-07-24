import json
import os

# ====== BASE PATH ======
BASE = r"C:\Users\vaibh\OneDrive\Desktop\CROWD_RISK_PROJECT\datasets\coco2017"

COCO_JSON = os.path.join(BASE, "annotations", "instances_val2017.json")
IMAGES_DIR = os.path.join(BASE, "images", "val2017")
LABELS_DIR = os.path.join(BASE, "labels", "val2017")

os.makedirs(LABELS_DIR, exist_ok=True)

# ====== LOAD COCO JSON ======
with open(COCO_JSON, "r") as f:
    coco = json.load(f)

# Get person category ID
person_id = None
for category in coco["categories"]:
    if category["name"] == "person":
        person_id = category["id"]
        break

if person_id is None:
    print("❌ Person class not found.")
    exit()

# Create image ID lookup
image_dict = {}
for img in coco["images"]:
    image_dict[img["id"]] = img

print("Converting annotations...")

# Process annotations
for ann in coco["annotations"]:
    if ann["category_id"] != person_id:
        continue

    img_info = image_dict[ann["image_id"]]
    file_name = img_info["file_name"]
    width = img_info["width"]
    height = img_info["height"]

    x, y, w, h = ann["bbox"]

    # Convert to YOLO format
    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height
    w /= width
    h /= height

    label_file = os.path.join(LABELS_DIR, file_name.replace(".jpg", ".txt"))

    with open(label_file, "a") as f:
        f.write(f"0 {x_center} {y_center} {w} {h}\n")

print("✅ Conversion completed successfully.")