import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file
from dataset import ViolenceDataset
from model import get_model

CLASS_NAMES = ["NonViolence", "Violence"]

def load_model(model_path: str | Path, device: torch.device):
    model = get_model(num_classes=2, pretrained=False)
    state_dict = load_file(str(model_path))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_with_model(model, device, video_path: str | Path, num_frames: int = 16):
    dataset = ViolenceDataset([(str(video_path), 0)], num_frames=num_frames)
    frames, _, _ = dataset[0]
    frames = frames.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(frames)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = int(probabilities.argmax().item())

    result = {
        "video_path": str(video_path),
        "predicted_label": CLASS_NAMES[predicted_class],
        "confidence": float(probabilities[predicted_class].item()),
        "probabilities": {
            class_name: float(probabilities[idx].item())
            for idx, class_name in enumerate(CLASS_NAMES)
        },
    }
    return result

def predict_video(video_path: str | Path, model_path: str | Path, num_frames: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    return predict_with_model(model, device, video_path, num_frames=num_frames)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument(
        "--model-path",
        default="models/best_violence_model.safetensors",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    args = parser.parse_args()

    result = predict_video(args.video_path, args.model_path, args.num_frames)
    print(f"Video: {result['video_path']}")
    print(f"Prediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("Class probabilities:")
    for class_name, probability in result["probabilities"].items():
        print(f"  {class_name}: {probability:.2%}")



