import os
import sys
import shutil
import json
import cv2
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from datetime import datetime
from PIL import Image

BASE_DIR = r"C:\zsign-detection"
NEW_DATA_DIR = os.path.join(BASE_DIR, "Data", "newdata")
OLD_DATA_DIR = os.path.join(BASE_DIR, "Data", "final_data")
MERGED_DIR = os.path.join(BASE_DIR, "Data", "merged_data")
SAMPLE_DIR = os.path.join(BASE_DIR, "Data", "sample_verifications")

CLASS_MAP = {
    "hello": 0,
    "love": 1,
    "no": 2,
    "please": 3,
    "thanx": 4,
    "yes": 5,
    "bad": 6,
    "food": 7,
    "friend": 8,
    "good": 9,
    "help": 10,
    "pain": 11,
    "stop": 12,
    "water": 13,
    "welcome": 14,
    "where": 15
}

CLASS_NAMES = [
    "Hello", "IloveYou", "No", "Please", "Thanks", "Yes",
    "Bad", "Food", "Friend", "Good", "Help", "Pain",
    "Stop", "Water", "Welcome", "Where"
]

def voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    # Clamp coordinates to image boundaries
    xmin = max(0.0, min(float(xmin), float(img_w)))
    xmax = max(0.0, min(float(xmax), float(img_w)))
    ymin = max(0.0, min(float(ymin), float(img_h)))
    ymax = max(0.0, min(float(ymax), float(img_h)))
    
    x_center = ((xmin + xmax) / 2.0) / float(img_w)
    y_center = ((ymin + ymax) / 2.0) / float(img_h)
    width = (xmax - xmin) / float(img_w)
    height = (ymax - ymin) / float(img_h)
    
    return x_center, y_center, width, height

def parse_xml_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size_elem = root.find("size")
    img_w = int(size_elem.find("width").text)
    img_h = int(size_elem.find("height").text)
    
    yolo_lines = []
    objects_meta = []
    
    for obj in root.findall("object"):
        raw_name = obj.find("name").text.strip().lower()
        if raw_name not in CLASS_MAP:
            raise ValueError(f"Unknown class '{raw_name}' in {xml_path}")
        
        class_id = CLASS_MAP[raw_name]
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        
        xc, yc, w, h = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
        yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        objects_meta.append({
            "name": raw_name,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "voc_box": [xmin, ymin, xmax, ymax],
            "yolo_box": [xc, yc, w, h]
        })
        
    return img_w, img_h, yolo_lines, objects_meta

def main():
    print("=" * 60)
    print("Step 1: Auditing & Reading Data/newdata...")
    print("=" * 60)
    
    all_files = os.listdir(NEW_DATA_DIR)
    xml_files = sorted([f for f in all_files if f.lower().endswith(".xml")])
    jpg_files = sorted([f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    
    print(f"Total XML annotations: {len(xml_files)}")
    print(f"Total JPG images: {len(jpg_files)}")
    
    # Check 1-to-1 match
    xml_stems = {os.path.splitext(f)[0] for f in xml_files}
    jpg_stems = {os.path.splitext(f)[0] for f in jpg_files}
    assert xml_stems == jpg_stems, "Mismatch between XMLs and Images!"
    
    # Group items by class and detect burst sessions
    # Each filename is: WIN_YYYYMMDD_HH_MM_SS_Pro.xml
    class_to_items = defaultdict(list)
    parsed_data = {}
    
    for xml_f in xml_files:
        stem = os.path.splitext(xml_f)[0]
        jpg_f = stem + ".jpg"
        xml_p = os.path.join(NEW_DATA_DIR, xml_f)
        jpg_p = os.path.join(NEW_DATA_DIR, jpg_f)
        
        w, h, ylines, ometa = parse_xml_annotation(xml_p)
        
        parts = stem.replace("_Pro", "").split("_")
        # WIN, YYYYMMDD, HH, MM, SS
        dt = datetime.strptime(f"{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}", "%Y%m%d_%H_%M_%S")
        
        item = {
            "stem": stem,
            "xml_file": xml_f,
            "jpg_file": jpg_f,
            "datetime": dt,
            "width": w,
            "height": h,
            "yolo_lines": ylines,
            "meta": ometa,
            "primary_class": ometa[0]["name"]
        }
        parsed_data[stem] = item
        class_to_items[item["primary_class"]].append(item)
    
    print("\nAudited classes and counts:")
    for cname, items in sorted(class_to_items.items()):
        print(f"  {cname:10s} -> {CLASS_MAP[cname]:2d} ({CLASS_NAMES[CLASS_MAP[cname]]:8s}): {len(items)} images")
        
    print("\n" + "=" * 60)
    print("Step 2: Splitting new data by session into Train & Val...")
    print("=" * 60)
    
    new_train_items = []
    new_val_items = []
    
    session_report = {}
    
    for cname, items in sorted(class_to_items.items()):
        items_sorted = sorted(items, key=lambda x: x["datetime"])
        sessions = [[]]
        for i, it in enumerate(items_sorted):
            if i > 0 and (it["datetime"] - items_sorted[i-1]["datetime"]).total_seconds() > 60:
                sessions.append([])
            sessions[-1].append(it)
        
        # Session 1 -> Train, Session 2 -> Val
        s1 = sessions[0]
        s2 = sessions[1] if len(sessions) > 1 else []
        
        new_train_items.extend(s1)
        new_val_items.extend(s2)
        
        session_report[cname] = {
            "session_1_train": len(s1),
            "session_2_val": len(s2),
            "total": len(items)
        }
        print(f"  {cname:10s}: Session 1 (Train) = {len(s1)}, Session 2 (Val) = {len(s2)}")
        
    print(f"\nTotal new images in Train: {len(new_train_items)}")
    print(f"Total new images in Val:   {len(new_val_items)}")
    
    print("\n" + "=" * 60)
    print("Step 3: Creating Merged Dataset Structure...")
    print("=" * 60)
    
    # Clean output directories if they have prior contents
    subdirs = [
        "images/train", "images/val", "images/test", "images/val_new_only",
        "labels/train", "labels/val", "labels/test", "labels/val_new_only"
    ]
    for sub in subdirs:
        p = os.path.join(MERGED_DIR, sub)
        shutil.rmtree(p, ignore_errors=True)
        os.makedirs(p, exist_ok=True)
        
    # Helper to write new items with "new_" prefix
    def write_new_items(items, img_dest_dir, lbl_dest_dir):
        for it in items:
            dest_img_name = f"new_{it['jpg_file']}"
            dest_lbl_name = f"new_{it['stem']}.txt"
            
            src_img = os.path.join(NEW_DATA_DIR, it["jpg_file"])
            dst_img = os.path.join(img_dest_dir, dest_img_name)
            dst_lbl = os.path.join(lbl_dest_dir, dest_lbl_name)
            
            shutil.copy2(src_img, dst_img)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(it["yolo_lines"]) + "\n")

    # 1. Write new Train items
    write_new_items(new_train_items,
                    os.path.join(MERGED_DIR, "images", "train"),
                    os.path.join(MERGED_DIR, "labels", "train"))
                    
    # 2. Write new Val items (to images/val and images/val_new_only)
    write_new_items(new_val_items,
                    os.path.join(MERGED_DIR, "images", "val"),
                    os.path.join(MERGED_DIR, "labels", "val"))
    write_new_items(new_val_items,
                    os.path.join(MERGED_DIR, "images", "val_new_only"),
                    os.path.join(MERGED_DIR, "labels", "val_new_only"))
                    
    # 3. Copy Old Train data (120 images) into merged images/train
    old_train_img_dir = os.path.join(OLD_DATA_DIR, "train", "images")
    old_train_lbl_dir = os.path.join(OLD_DATA_DIR, "train", "labels")
    old_train_files = os.listdir(old_train_img_dir)
    for f in old_train_files:
        stem = os.path.splitext(f)[0]
        shutil.copy2(os.path.join(old_train_img_dir, f), os.path.join(MERGED_DIR, "images", "train", f))
        shutil.copy2(os.path.join(old_train_lbl_dir, stem + ".txt"), os.path.join(MERGED_DIR, "labels", "train", stem + ".txt"))
        
    # 4. Copy Old Test data (30 images) into merged images/val and images/test (frozen benchmark)
    old_test_img_dir = os.path.join(OLD_DATA_DIR, "test", "images")
    old_test_lbl_dir = os.path.join(OLD_DATA_DIR, "test", "labels")
    old_test_files = os.listdir(old_test_img_dir)
    for f in old_test_files:
        stem = os.path.splitext(f)[0]
        # into images/val
        shutil.copy2(os.path.join(old_test_img_dir, f), os.path.join(MERGED_DIR, "images", "val", f))
        shutil.copy2(os.path.join(old_test_lbl_dir, stem + ".txt"), os.path.join(MERGED_DIR, "labels", "val", stem + ".txt"))
        # into images/test (frozen baseline)
        shutil.copy2(os.path.join(old_test_img_dir, f), os.path.join(MERGED_DIR, "images", "test", f))
        shutil.copy2(os.path.join(old_test_lbl_dir, stem + ".txt"), os.path.join(MERGED_DIR, "labels", "test", stem + ".txt"))

    print("\nMerged Dataset Counts:")
    print(f"  images/train: {len(os.listdir(os.path.join(MERGED_DIR, 'images', 'train')))} images (120 old + {len(new_train_items)} new)")
    print(f"  labels/train: {len(os.listdir(os.path.join(MERGED_DIR, 'labels', 'train')))} labels")
    print(f"  images/val:   {len(os.listdir(os.path.join(MERGED_DIR, 'images', 'val')))} images (30 old + {len(new_val_items)} new)")
    print(f"  labels/val:   {len(os.listdir(os.path.join(MERGED_DIR, 'labels', 'val')))} labels")
    print(f"  images/test:  {len(os.listdir(os.path.join(MERGED_DIR, 'images', 'test')))} images (frozen baseline 30 images)")
    print(f"  images/val_new_only: {len(os.listdir(os.path.join(MERGED_DIR, 'images', 'val_new_only')))} images (new validation slice)")

    print("\n" + "=" * 60)
    print("Step 4: Drawing Spot-Check Samples with Bounding Boxes...")
    print("=" * 60)
    
    # Draw 1 sample per class (total 16 samples) to verify conversion
    shutil.rmtree(SAMPLE_DIR, ignore_errors=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    for cname, items in class_to_items.items():
        sample_item = items[0]
        img_p = os.path.join(NEW_DATA_DIR, sample_item["jpg_file"])
        img = cv2.imread(img_p)
        
        for obj in sample_item["meta"]:
            cid = obj["class_id"]
            c_label = obj["class_name"]
            xmin, ymin, xmax, ymax = [int(v) for v in obj["voc_box"]]
            
            # Draw box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # Draw label banner
            text = f"{c_label} (ID:{cid})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (xmin, max(0, ymin - 22)), (xmin + tw + 6, max(0, ymin)), (0, 255, 0), -1)
            cv2.putText(img, text, (xmin + 3, max(16, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            
        out_sample = os.path.join(SAMPLE_DIR, f"sample_{sample_item['meta'][0]['class_name']}.jpg")
        cv2.imwrite(out_sample, img)
        print(f"  Generated verification sample: {os.path.basename(out_sample)}")

    print("\n" + "=" * 60)
    print("Step 5: Writing YAML Configurations...")
    print("=" * 60)
    
    merged_dir_posix = MERGED_DIR.replace("\\", "/")
    
    # 1. dataset_merged.yaml
    merged_yaml_content = f"""# Merged Dataset Configuration (16 Sign Language Classes)
path: {merged_dir_posix}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(os.path.join(BASE_DIR, "dataset_merged.yaml"), "w", encoding="utf-8") as f:
        f.write(merged_yaml_content)
    print("  Created dataset_merged.yaml")
    
    # 2. dataset_baseline_test.yaml
    baseline_yaml_content = f"""# Frozen Baseline Test Set (30 original images)
path: {merged_dir_posix}
train: images/train
val: images/test
test: images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(os.path.join(BASE_DIR, "dataset_baseline_test.yaml"), "w", encoding="utf-8") as f:
        f.write(baseline_yaml_content)
    print("  Created dataset_baseline_test.yaml")
    
    # 3. dataset_newdata_val.yaml
    newdata_val_content = f"""# New Data Validation Slice (125 new images)
path: {merged_dir_posix}
train: images/train
val: images/val_new_only
test: images/val_new_only

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(os.path.join(BASE_DIR, "dataset_newdata_val.yaml"), "w", encoding="utf-8") as f:
        f.write(newdata_val_content)
    print("  Created dataset_newdata_val.yaml")
    
    # 4. Model Architecture: custom_yolov5s_16cls.yaml
    model_cfg = f"""# YOLOv5s model config for 16 classes
nc: {len(CLASS_NAMES)}  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, BottleneckCSP, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
"""
    with open(os.path.join(BASE_DIR, "yolov5", "models", "custom_yolov5s_16cls.yaml"), "w", encoding="utf-8") as f:
        f.write(model_cfg)
    print("  Created yolov5/models/custom_yolov5s_16cls.yaml")
    
    # 5. Domain Hyperparameters: hyp.sign_language.yaml
    hyp_cfg = """# Hyperparameters optimized for fine-tuning sign language gestures
# fliplr set to 0.0 to prevent horizontal flip distorting hand orientation
lr0: 0.005
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.2
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.0  # Avoid horizontal flips for sign language orientation
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
"""
    with open(os.path.join(BASE_DIR, "yolov5", "data", "hyps", "hyp.sign_language.yaml"), "w", encoding="utf-8") as f:
        f.write(hyp_cfg)
    print("  Created yolov5/data/hyps/hyp.sign_language.yaml")
    
    # Write audit and summary report
    audit_report = {
        "timestamp": datetime.now().isoformat(),
        "total_new_images": len(jpg_files),
        "total_new_xmls": len(xml_files),
        "classes_count": len(CLASS_NAMES),
        "class_mapping": CLASS_MAP,
        "class_names": CLASS_NAMES,
        "per_class_session_split": session_report,
        "merged_dataset": {
            "train_images": len(os.listdir(os.path.join(MERGED_DIR, "images", "train"))),
            "val_images": len(os.listdir(os.path.join(MERGED_DIR, "images", "val"))),
            "frozen_test_images": len(os.listdir(os.path.join(MERGED_DIR, "images", "test"))),
            "val_new_only_images": len(os.listdir(os.path.join(MERGED_DIR, "images", "val_new_only")))
        }
    }
    with open(os.path.join(BASE_DIR, "Data", "audit_report.json"), "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print("  Created Data/audit_report.json")
    print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
    main()
