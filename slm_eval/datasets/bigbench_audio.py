# Copyright 2025 Xiaomi Corporation.
from datasets import load_dataset, Audio
from pathlib import Path
import json
import os


class BigBenchAudioDataset:
    
    def __init__(self, sample_rate=24000):
        self.ds = load_dataset("ArtificialAnalysis/big_bench_audio")['train']
        self.data = self.ds.cast_column("audio", Audio(sampling_rate=sample_rate))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __iter__(self):
        return iter(self.data)
