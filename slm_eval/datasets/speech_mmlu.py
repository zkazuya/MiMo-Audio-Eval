# Copyright 2025 Xiaomi Corporation.
import os
import json
import subprocess
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset, get_dataset_config_names, load_from_disk
from huggingface_hub import snapshot_download


class SpeechMMLUDataset:

    def __init__(self, split="all"):
        base_dir = Path(os.getcwd())
        self.data_path = base_dir / "data" / "SpeechMMLU"
        self.lang = "en"
        self.num_voice_ids = 250
        
        if split == "all":
            self.include_all = True
        else:
            self.include_all = False
            self.subjects_to_include = [s.strip() for s in split.split(',')]
        
        self.data_by_subject, self.few_shot_prompts = self.load_data()
        
        self.data = [item for items in self.data_by_subject.values() for item in items]
        self.subjects = list(self.data_by_subject.keys())
        
        print(f"Loaded dataset: {len(self.data)} items across {len(self.subjects)} subjects")

    def load_data(self):  
        available_subjects = get_dataset_config_names(str(self.data_path))
        
        if self.include_all:
            target_subjects = sorted(available_subjects)
        else:
            target_subjects = []
            for subject in self.subjects_to_include:
                if subject in available_subjects:
                    target_subjects.append(subject)
                else:
                    print(f"Warning: Subject '{subject}' not found. Available: {sorted(available_subjects)}")
        
        if not target_subjects:
            raise ValueError(f"No valid subjects found. Available: {sorted(available_subjects)}")
        
        print(f"Loading subjects: {target_subjects}")
        
        data_by_subject = defaultdict(list)
        for subject in target_subjects:
            print(f"Loading subject: {subject}")
            dataset = load_dataset(str(self.data_path), subject)['train']
            for item in dataset:
                data_item = {
                    "id": item["id"],
                    "subject": item["subject"],
                    "question_and_choices": {
                        "text": item["question_text"],
                        "speech": self.data_path / item["question_audio"],
                    },
                    "voice_id": item["voice_id"],
                    "answer": item["answer"],
                }
                data_by_subject[subject].append(data_item)
            
            print(f"Loaded {subject}: {len(data_by_subject[subject])} items")

        few_shot_prompts_file_path = self.data_path / "few_shot_prompts.json"
        
        with open(few_shot_prompts_file_path, "r") as f:
            few_shot_prompts_info = json.load(f)
        
        few_shot_prompts = defaultdict(lambda: defaultdict(dict))
        for subject in target_subjects:
            meta_info = few_shot_prompts_info[subject]
            
            for _voice_id in range(self.num_voice_ids):
                voice_id = f"english_prompt_{str(_voice_id+1).zfill(3)}"
                few_shot_prompts[subject][voice_id] = {
                    "beginning": {
                        "text": meta_info["beginning"],
                        "speech": self.data_path / "audio" / subject / "prompts" / f"{subject}_{voice_id}" / f"{subject}_beginning.mp3",
                    },
                    "examples": [
                        {
                            "question_and_choices": {
                                "text": example["question"],
                                "speech": self.data_path / "audio" / subject / "prompts" / f"{subject}_{voice_id}" / f"{subject}_{i}_question_and_choices.mp3",
                            },
                            "answer": {
                                "text": example["answer"],
                                "speech": self.data_path / "audio" / subject / "prompts" / f"{subject}_{voice_id}" / f"{subject}_{i}_answer.mp3",
                            }
                        }
                        for i, example in enumerate(meta_info["examples"])
                    ]
                }
        
        return dict(data_by_subject), few_shot_prompts

    def get_few_shot_prompts(self, subject, voice_id):
        return self.few_shot_prompts[subject][voice_id]