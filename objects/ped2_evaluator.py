import os
import re

class Ped2Evaluator:
    def __init__(self, gt_m_file="UCSDped2.m"):
        self.gt = self._load_ground_truth(gt_m_file)

    def _load_ground_truth(self, m_path):
        gt = {}
        current_index = 0
        with open(m_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                match = re.search(r"gt_frame\s*=\s*\[([0-9:\s]+)\];", line)
                if match:
                    raw_range = match.group(1).strip()
                    frames = []
                    for token in raw_range.split():
                        if ':' in token:
                            parts = list(map(int, token.split(':')))
                            if len(parts) == 2:
                                start, end = parts
                                frames.extend(range(start, end + 1))
                        else:
                            frames.append(int(token))
                    gt[current_index] = [f - 1 for f in frames]  # Convert to 0-indexed
                    current_index += 1
        return gt

    def generate_labels(self, paths):
        labels = []
        for path in paths:
            parts = str(path).split(os.sep)
            video_folder = next((p for p in parts if p.lower().startswith("test") and p[4:].isdigit()), None)
            if video_folder is None:
                raise ValueError(f"Could not determine test video folder from path: {path}")
            video_idx = int(video_folder.replace("Test", "")) - 1
            frame_str = os.path.splitext(os.path.basename(path))[0]  # e.g., 00160
            frame_idx = int(frame_str) - 1  # convert to 0-indexed
            label = 1 if frame_idx in self.gt.get(video_idx, []) else 0
            labels.append(label)
        return labels
    