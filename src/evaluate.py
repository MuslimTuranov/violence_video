import argparse
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from dataset import collect_video_paths
from inference import load_model, predict_with_model

CLASS_TO_LABEL = {"NonViolence": 0, "Violence": 1}

def evaluate_dataset(dataset_dir, model_path, num_frames):
    samples = collect_video_paths(dataset_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    y_true = []
    y_pred = []
    predictions = []

    for video_path, label in samples:
        result = predict_with_model(model, device, video_path, num_frames=num_frames)
        predicted_label = CLASS_TO_LABEL[result["predicted_label"]]
        y_true.append(label)
        y_pred.append(predicted_label)
        predictions.append(
            {
                "video_path": str(video_path),
                "target": label,
                "prediction": predicted_label,
                "confidence": result["confidence"],
            }
        )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["NonViolence", "Violence"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "num_videos": len(samples),
        "predictions": predictions,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path containing Violence/ and NonViolence/ folders to evaluate",
    )
    parser.add_argument(
        "--model-path",
        default="models/best_violence_model.safetensors",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    metrics = evaluate_dataset(args.dataset_dir, args.model_path, args.num_frames)
    formatted = json.dumps(metrics, indent=2)
    print(formatted)
    
    if args.output:
        Path(args.output).write_text(formatted, encoding="utf-8")


if __name__ == "__main__":
    main()
