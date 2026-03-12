# Violence Video Classification

PyTorch project for binary video classification on the Real Life Violence Situations dataset.

## Results

- Backbone: pretrained `torchvision.models.video.r3d_18`
- Output format: `.safetensors`
- Validation accuracy: `94.63%`
- External video accuracy: `83.33%` (`5/6`)

The model works well on explicit violent actions and misses some ambiguous threat-like scenes.

## Project files

- `src/train.py` - training
- `src/evaluate.py` - evaluation on labeled folders
- `src/inference.py` - single-video inference
- `src/dataset.py` - frame extraction and preprocessing
- `src/model.py` - model definition
- `models/` - trained model and metric files

## Dataset layout

```text
videodataset/
  NonViolence/
  Violence/
```

External evaluation uses the same structure:

```text
test_videos/
  NonViolence/
  Violence/
```

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py --dataset-dir videodataset --output-dir models --epochs 12 --batch-size 4 --num-frames 16 --train-backbone --num-workers 0
```

Saved artifacts:

- `models/best_violence_model.safetensors`
- `models/best_violence_model_config.json`
- `models/best_val_metrics.json`
- `models/data_split.json`

## Evaluate

Validation-style evaluation:

```bash
python src/evaluate.py --dataset-dir videodataset --model-path models/best_violence_model.safetensors --output models/metrics_full_dataset.json
```

External videos:

```bash
python src/evaluate.py --dataset-dir test_videos --model-path models/best_violence_model.safetensors --output models/metrics_real_videos.json
```

External test summary:

- NonViolence: `3/3` correct
- Violence: `2/3` correct
- Confusion matrix:

```text
[
  [3, 0],
  [1, 2]
]
```

## Inference

```bash
python src/inference.py test_videos/Violence/example.mp4 --model-path models/best_violence_model.safetensors
```

## Training setup

- Architecture: `r3d_18`
- Pretrained weights: `R3D_18_Weights.DEFAULT`
- Classes: `2`
- Frames per video: `16`
- Frame size: `112x112`
- Optimizer: `AdamW`
- Loss: `CrossEntropyLoss`
- Scheduler: `ReduceLROnPlateau`

## Notes

- Training was done on CPU, so it is slow without GPU acceleration.
- Some dataset videos produce ffmpeg/OpenCV decode warnings.
- Commit the code, `.safetensors` model, metric JSON files, and either the external test videos or their sources/labels.
