# RacketPose: Monocular Racket Pose Estimation Dataset

This repository provides the dataset and benchmark for monocular racket pose estimation.

The dataset contains RGB images, racket bounding boxes, and 3D racket pose annotations, including racket center coordinates and racket surface normal vectors.

---

# Dataset Structure

```text
data/
  imgs/
    <sequence_name>/
      *.jpg

  boxes/
    <sequence_name>/
      bounding box files

  labels/
    train/all.csv
    val/all.csv
    test/all.csv
```

---

# Annotation Format

Each split contains an `all.csv` file.

Example columns:

```text
filename,
center_x, center_y, center_z,
normal_x, normal_y, normal_z,
label
```

Where:

* `filename`: relative image path
* `center_x, center_y, center_z`: 3D racket center coordinates in the camera coordinate system
* `normal_x, normal_y, normal_z`: normalized racket surface normal vector
* `label`: racket category ID

---

# Coordinate System

The dataset uses the following camera-centered coordinate system:

* `x`: horizontal right
* `y`: vertical up
* `z`: forward from the camera

All 3D annotations are represented in meters.

---

# Dataset Splits

The dataset is divided into:

* `train`
* `val`
* `test`

Each split contains independent annotation files.

---

# Representative Sample Dataset

For reviewers and quick inspection, a representative sample dataset is also provided.

The sample dataset:

* preserves the original directory structure
* contains images, labels, and bounding boxes
* is created using stratified random sampling from the full dataset

---

# Intended Usage

This dataset is designed for:

* monocular racket pose estimation
* racket orientation estimation
* 3D sports object understanding
* ROI-based pose regression research
* context-aware pose estimation

---

# Training 
```
python main_pose_roi.py --train  (for global-local)
python main_global.py --train (for global-only) 
python main_pose_person_racket.py --train (for person-local) 
python main_pose_vit_global.py --train (for vit) 
python main_global_swin.py --train (for swin) 
```
The detector weight need to be put under folder "weight/" 
* https://pan.baidu.com/s/1Kpvqd-sTTyjze13gJn1m3g?pwd=3fc4



# Reference / Acknowledgements

This project references or is inspired by the following repositories and projects:

* https://github.com/jahongir7174/YOLOv11-pt
* https://github.com/ultralytics/ultralytics
* https://github.com/jahongir7174/YOLOv8-pt

---

# Notes

* Bounding box files are optional for training if labels already contain valid image paths.
* The provided sample dataset is intended only for dataset inspection and reproducibility checking.
* The full dataset is required for reproducing benchmark results.
