from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.models.video import R3D_18_Weights


class VideoNormalize:
    def __init__(self):
        preset = R3D_18_Weights.DEFAULT.transforms()
        self.mean = torch.tensor(preset.mean).view(3, 1, 1, 1)
        self.std = torch.tensor(preset.std).view(3, 1, 1, 1)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        return (frames - self.mean) / self.std


def build_video_transform():
    return VideoNormalize()


def collect_video_paths(dataset_dir: str | Path):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        project_root = Path(__file__).resolve().parents[1]
        candidate = project_root / dataset_dir
        if candidate.exists():
            dataset_dir = candidate

    class_to_label = {
        "NonViolence": 0,
        "Violence": 1,
    }

    samples = []
    for class_name, label in class_to_label.items():
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing dataset directory: {class_dir}")
        for video_path in sorted(class_dir.glob("*.mp4")):
            samples.append((video_path, label))
    if not samples:
        raise FileNotFoundError(f"No .mp4 videos found in {dataset_dir}")

    return samples


class ViolenceDataset(Dataset):
    def __init__(
        self,
        samples,
        num_frames: int = 16,
        frame_size: int = 112,
        transform=None,
    ):
        self.samples = [(str(video_path), label) for video_path, label in samples]
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform or build_video_transform()

    def __len__(self):
        return len(self.samples)

    def _read_frame(self, capture: cv2.VideoCapture, frame_index: int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.frame_size, self.frame_size))
        return frame

    def extract_frames(self, video_path: str | Path):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open: {video_path}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = max(total_frames, 1)
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        last_valid_frame = None
        for frame_index in frame_indices:
            frame = self._read_frame(capture, int(frame_index))
            if frame is None:
                if last_valid_frame is None:
                    frame = np.zeros(
                        (self.frame_size, self.frame_size, 3), dtype=np.uint8
                    )
                else:
                    frame = last_valid_frame.copy()
            else:
                last_valid_frame = frame
            frames.append(frame)
        capture.release()
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        if self.transform:
            frames = self.transform(frames)
        return frames

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self.extract_frames(video_path)
        return frames, torch.tensor(label, dtype=torch.long), video_path
