# Copyright 2025 Xiaomi Corporation.
from datasets import load_dataset


class InstructTTSDataset:
    
    def __init__(self, split="zh"):
        self.ds = load_dataset("CaasiHUANG/InstructTTSEval", split=split)
        self.lang = split
    
    def __len__(self):
        if hasattr(self.ds, '__len__'):
            return len(self.ds)
        return 0
    
    def __getitem__(self, idx):
        return self.ds[idx]
    
    def __iter__(self):
        return iter(self.ds)
