# Import native packages
import os

# Import external packages
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class VideoDataset(Dataset):
    """
    A PyTorch Dataset that returns temporal sequences of frames from subdirectories,
    where each subdirectory represents a video composed of individual image frames.
    Sampling and frame loading are separated for clean, stateless access.
    """

    def __init__(self, directory, temporal=3, resize=(384, 256), transform=None):
        self.directory = directory                            # Root folder containing video subfolders
        self.temporal = temporal                              # Number of frames in a temporal clip
        self.resize = resize                                  # Resize shape: (H, W)

        # Use default transform if none provided
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor()
        ])

        # Build index of (subdirectory, start_index) pairs
        self.sample_index_list = []                           # List of tuples (video_path, start_idx)
        self.subdirectories = sorted([
            os.path.join(self.directory, d)
            for d in os.listdir(self.directory)
            if os.path.isdir(os.path.join(self.directory, d))
        ])

        for subdirectory in self.subdirectories:
            frames = sorted([
                os.path.join(subdirectory, f)
                for f in os.listdir(subdirectory)
                if os.path.isfile(os.path.join(subdirectory, f)) and f[-3:] in ['png', 'jpg', 'tif']
            ])
            num_frames = len(frames)
            for i in range(num_frames - self.temporal + 1):
                self.sample_index_list.append((frames[i:i + self.temporal]))

    def __len__(self):
        return len(self.sample_index_list)

    def __getitem__(self, idx):
        frame_paths = self.sample_index_list[idx]
        frames = []

        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            image_tensor = self.transform(image)
            frames.append(image_tensor)

        frame_tensor = torch.stack(frames, dim=1)  # Shape: (C, T, H, W)
        return frame_tensor, frame_paths
