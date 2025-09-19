# Copyright 2025 Xiaomi Corporation.
import os
from typing import List, Tuple, Union, Iterable, Dict
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class GeneralS2SDataset:
    
    def __init__(self, split: Optional[str] = None) -> None:
        self.base_dir = Path(os.getcwd())
        if split == "ZH001SpeedConversion_speed1.0to0.5":
            self.scp_path = self.base_dir / "data/speed_conversion/ZH001SpeedConversion_speed1.0to0.5.scp"
            self.dataset_lang = "zh"
        elif split == "ZH001SpeedConversion_speed1.0to1.5":
            self.scp_path = self.base_dir / "data/speed_conversion/ZH001SpeedConversion_speed1.0to1.5.scp"
            self.dataset_lang = "zh"
        elif split == "ZH001SpeedConversion_speed0.5to1.0":
            self.scp_path = self.base_dir / "data/speed_conversion/ZH001SpeedConversion_speed0.5to1.0.scp"
            self.dataset_lang = "zh"
        elif split == "ex01_happy2sad":
            self.scp_path = self.base_dir / "data/expresso/ex01_happy2sad.scp"
            self.dataset_lang = "en"
        elif split == "ex03_default2whisper":
            self.scp_path = self.base_dir / "data/expresso/ex03_default2whisper.scp"
            self.dataset_lang = "en"
        elif split == "ex02_default2laughing":
            self.scp_path = self.base_dir / "data/expresso/ex02_default2laughing.scp"
            self.dataset_lang = "en"
        else:
            raise ValueError(f"Invalid split: {split}")
        self.delimiter = "|"
        
        self.input_paths, self.target_paths = self._load_scp_file()
        self.dataset_name = split
        
    def _load_scp_file(self) -> Tuple[List[str], List[str]]:
        if not os.path.exists(self.scp_path):
            raise FileNotFoundError(f"SCP file not found: {self.scp_path}")
            
        input_paths = []
        target_paths = []
        
        with open(self.scp_path, 'r') as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    input_path, target_path = line.split(self.delimiter)
                    if not os.path.exists(input_path):
                        logger.warning(f"Input file not found (line {line_idx}): {input_path}")
                        continue
                    if not os.path.exists(target_path):
                        logger.warning(f"Target file not found (line {line_idx}): {target_path}")
                        continue
                        
                    input_paths.append(input_path)
                    target_paths.append(target_path)
                    
                except ValueError:
                    logger.warning(f"Invalid line format at line {line_idx}: {line}")
                    continue
                    
        if not input_paths:
            raise RuntimeError(f"No valid pairs found in SCP file: {self.scp_path}")
            
        logger.info(f"Loaded {len(input_paths)} valid pairs from {self.scp_path}")
        return input_paths, target_paths
        
    def __len__(self) -> int:
        return len(self.input_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            "source": str(self.base_dir / self.input_paths[idx]),
            "target": str(self.base_dir / self.target_paths[idx])
        }
    
    def __iter__(self) -> Iterable[Dict[str, str]]:
        for input_path, target_path in zip(self.input_paths, self.target_paths):
            yield {
                "source": str(self.base_dir / input_path),
                "target": str(self.base_dir / target_path)
            }
