# Copyright 2025 Xiaomi Corporation.
import torch
import os
from pathlib import Path
from datasets import load_dataset, Audio


class MMAUProDataset:

    def __init__(self, sample_rate=24000):
        self.data = load_dataset("gamma-lab-umd/MMAU-Pro")
        self.data = self.data['test']

        def convert_audio_paths_to_absolute(example):
            path_root = Path(os.getcwd()) / "data" / "MMAU-Pro"
            example['audio_path'] = [str(path_root / path) for path in example['audio_path']]
            return example
            
        self.data = self.data.map(convert_audio_paths_to_absolute)

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        for item in self.data:
            yield item