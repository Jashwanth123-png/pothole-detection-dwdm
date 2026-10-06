"""Inspect the dataset splits and class distribution."""
import os
from pathlib import Path

root = Path(__file__).parent

for split in ['train', 'val', 'test']:
    img_dir = root / 'data' / 'processed' / 'split' / split / 'images'
    lbl_dir = root / 'data' / 'processed' / 'split' / split / 'labels'
    imgs = list(img_dir.glob('*')) if img_dir.exists() else []
    lbls = list(lbl_dir.glob('*.txt')) if lbl_dir.exists() else []
    print(f"{split}: {len(imgs)} images, {len(lbls)} labels")

# Class counts in train
label_dir = root / 'data' / 'processed' / 'split' / 'train' / 'labels'
class_counts = {}
for f in label_dir.glob('*.txt'):
    with open(f) as file:
        for line in file:
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                class_counts[cls] = class_counts.get(cls, 0) + 1

print('\nClass distribution in train labels:')
for cls, count in sorted(class_counts.items()):
    print(f'  Class {cls}: {count} objects')

# Sample a few label files
print('\nSample label files:')
sample_files = list(label_dir.glob('*.txt'))[:5]
for f in sample_files:
    print(f'  {f.name}:')
    with open(f) as file:
        for line in file:
            print(f'    {line.rstrip()}')

# Check dataset.yaml
yaml_path = root / 'data' / 'processed' / 'split' / 'dataset.yaml'
print(f'\ndataset.yaml exists: {yaml_path.exists()}')
if yaml_path.exists():
    print(yaml_path.read_text())

# Check what classes exist in the RDD2022 dataset labels folder
rdd_label_dir = root / 'dataset' / 'rdd2022' / 'India' / 'train' / 'labels'
rdd_class_counts = {}
for f in rdd_label_dir.glob('*.txt'):
    with open(f) as file:
        for line in file:
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                rdd_class_counts[cls] = rdd_class_counts.get(cls, 0) + 1

print(f'\nRDD2022 label directory: {rdd_label_dir}')
print(f'RDD2022 label files: {sum(1 for _ in rdd_label_dir.glob("*.txt"))}')
print('Class distribution in RDD2022 India/train labels:')
for cls, count in sorted(rdd_class_counts.items()):
    print(f'  Class {cls}: {count} objects')
