# Copyright 2025 Xiaomi Corporation.
from .librispeech import LibriSpeechDataset
from .aishell1 import AiShell1Dataset
from .seedtts import SeedTTSDataset
from .speech_mmlu import SpeechMMLUDataset
from .instruct_tts import InstructTTSDataset
from .esd import ESDDataset
from .general_s2s import GeneralS2SDataset
from .mmsu import MMSUDataset
from .mmau import MMAUDataset
from .mmau_pro import MMAUProDataset
from .mmar import MMARDataset
from .multi_challenge import MultiChallengeDataset
from .bigbench_audio import BigBenchAudioDataset
from datasets import load_dataset, Audio


def get_dataset(dataset, split, sample_rate=24000):
    if dataset == "librispeech":
        return LibriSpeechDataset(split)
    if dataset == "aishell1":
        return AiShell1Dataset(split)
    if dataset == "seedtts":
        return SeedTTSDataset(split)
    if dataset == "speechmmlu":
        return SpeechMMLUDataset(split)
    if dataset == "instruct_tts":
        return InstructTTSDataset(split)
    if dataset == "esd":
        return ESDDataset(split)
    if dataset == "general_s2s":
        return GeneralS2SDataset(split)
    if dataset == "mmsu":
        return MMSUDataset()
    if dataset == "mmau":
        return MMAUDataset()
    if dataset == "mmau_pro":
        return MMAUProDataset()
    if dataset == "mmar":
        return MMARDataset()
    if dataset == "multi_challenge":
        return MultiChallengeDataset()
    if dataset == "bigbench_audio":
        return BigBenchAudioDataset(sample_rate)
    if dataset == "dynamic_superb_s2s":
        dataset = load_dataset(split, split="test")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
        dataset = dataset.cast_column("label", Audio(sampling_rate=sample_rate))
        return (dataset, split, sample_rate)