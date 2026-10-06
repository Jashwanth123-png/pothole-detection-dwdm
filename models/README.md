# Models Directory

This directory stores trained YOLO model weights.

## Expected File

```
models/
    pothole_yolo.pt   ← Trained YOLO model (place here after training)
```

## How to Get the Model

### Option 1: Train from Scratch (Requires RDD2022 Dataset)

```bash
# Step 1: Prepare RDD2022 dataset
python scripts/prepare_dataset.py

# Step 2: Train the YOLO model
python ml/train_yolo.py
```

This will save the trained model at `models/pothole_yolo.pt`.

### Option 2: Use a Pre-Trained Model

If you have a pre-trained YOLOv8 pothole detection model:

1. Place the `.pt` file here as `pothole_yolo.pt`
2. Update `model.yolo_model_path` in `config.yaml` if needed

### Demo Mode

If no model is available, the dashboard runs in **DEMO MODE** using synthetic data.
Demo mode clearly labels all visualizations as `DEMO DATA — NOT RDD2022 RESULTS`.
