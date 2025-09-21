# Copyright 2025 Xiaomi Corporation.
import torch
from datasets import load_dataset, Audio
import os
import json


class MultiChallengeDataset:
    
    def __init__(self):
        data_root = os.getcwd()
        cache_dir = os.path.join(data_root, "data/multi_challenge")
        self.data_root = cache_dir
        save_path = os.path.join(cache_dir, "data.jsonl")

        self.data = self.load_dataset(save_path)
        self.lang = "en"

        self.post_process()

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        for item in self.data:
            yield item
    
    def post_process(self):
        length = len(self.data)
        for i in range(length):
            num_turns = len(self.data[i]["speech_dialogue"])
            for j in range(num_turns):
                self.data[i]["speech_dialogue"][j]["content"] = os.path.join(self.data_root, self.data[i]["speech_dialogue"][j]["content"].lstrip("./"))

    def load_dataset(self, save_path):
        data_list = []
        readlines = open(save_path, "r").readlines()

        for i, line in enumerate(readlines):
            json_dict = json.loads(line.strip())
            data_list.append(json_dict)

        return data_list