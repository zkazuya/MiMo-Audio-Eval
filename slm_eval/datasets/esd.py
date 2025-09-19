# Copyright 2025 Xiaomi Corporation.
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class ESDDataset:

    def __init__(self, split: Optional[str] = None) -> None:
        self.base_dir = Path(os.getcwd())
        self.data_file = self.base_dir / "data/ESD/VoiceConversion.tsv"
        if not self.data_file.exists():
            raise FileNotFoundError(f"TSV data file not found: {self.data_file}")

        self.split = split
        self.data: List[Dict] = []

        self._transcript_cache: Dict[str, Dict[str, str]] = {}

        self._load_data()

    def _get_target_text(self, target_path: str) -> str:
        wav_path = Path(target_path)
        if not wav_path.suffix.lower() == ".wav":
            return ""

        speaker_dir = wav_path.parent.parent        # .../0015
        speaker_id  = speaker_dir.name              # "0015"
        utt_id      = wav_path.stem                 # "0015_001062"

        if speaker_id not in self._transcript_cache:
            transcript_path = speaker_dir / f"{speaker_id}.txt"
            mapping: Dict[str, str] = {}
            if transcript_path.exists():
                with transcript_path.open(encoding="utf-8") as f:
                    for line in f:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) >= 2:
                            mapping[parts[0]] = parts[1]
            
            self._transcript_cache[speaker_id] = mapping

        return self._transcript_cache[speaker_id].get(utt_id, "")

    def _load_data(self) -> None:
        with self.data_file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                prompt_list: List[Dict[str, str]] = []
                prompt_field = row.get("prompt", "").strip()

                if prompt_field:
                    for triple_str in prompt_field.split("|"):
                        parts = triple_str.split("#")
                        if len(parts) != 2:
                            continue
                        src_p, tgt_p = parts
                        src_p = str(self.base_dir / "data" / src_p)
                        tgt_p = str(self.base_dir / "data" / tgt_p)
                        prompt_list.append({
                            "source": src_p,
                            "target": tgt_p,
                            "target_text": self._get_target_text(tgt_p),
                        })

                example = {
                    "id": row.get("source_id", ""),
                    "source_id": row.get("source_id", ""),
                    "source": str(self.base_dir / "data" / row.get("source_path", "")),
                    "target_id": row.get("target_id", ""),
                    "target": str(self.base_dir / "data" / row.get("target_path", "")),
                    "prompt": prompt_list,
                }
                self.data.append(example)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]

    def __iter__(self) -> Iterable[Dict]:
        for idx, example in enumerate(self.data):
            if not example.get("id"):
                example["id"] = str(idx)
            yield example

