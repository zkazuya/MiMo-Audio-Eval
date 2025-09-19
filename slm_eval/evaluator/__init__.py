# Copyright 2025 Xiaomi Corporation.
from .asr_evaluator import ASREvaluator
from .tts_evaluator import TTSEvaluator
from .speechmmlu_evaluator import SpeechMMLUEvaluator
from .dynamic_superb_s2s_evaluator import DynamicSuperbS2SEvaluator
from .voice_conversion_s2s_evaluator import VoiceConversionS2SEvaluator
from .general_s2s_evaluator import GeneralS2SEvaluator
from .instruct_tts_evaluator import InstructTTSEvaluator
from .mmsu_evaluator import MMSUEvaluator
from .mmau_evaluator import MMAUEvaluator
from .mmar_evaluator import MMAREvaluator
from .multi_challenge_evaluator import MultiChallengeEvaluator
from .bigbench_audio_evaluator import BigBenchAudioEvaluator
from .mmau_pro_evaluator import MMAUProEvaluator


def get_evaluator(task, model, dataset, n_few_shots=0, device=None, model_type='base', thinking=False):
    print(f"task: {task}")
    if task == "asr":
        return ASREvaluator(model, dataset, model_type=model_type, device=device)
    if task == "tts":
        return TTSEvaluator(model, dataset, model_type=model_type, device=device)
    if task.startswith("speechmmlu"):
        return SpeechMMLUEvaluator(model, dataset, task, n_few_shots, device=device)
    if task == "dynamic_superb_s2s":
        return DynamicSuperbS2SEvaluator(model, dataset, n_few_shots)
    if task == "voice_conversion_s2s":
        return VoiceConversionS2SEvaluator(model, dataset, n_few_shots)
    if task == "general_s2s":
        return GeneralS2SEvaluator(model, dataset, n_few_shots)
    if task == "instruct_tts":
        return InstructTTSEvaluator(model, dataset, model_type=model_type, device=device)
    if task == "mmsu":
        return MMSUEvaluator(model, dataset, model_type=model_type, n_few_shots=n_few_shots, thinking=thinking)
    if task == "mmau":
        return MMAUEvaluator(model, dataset, model_type=model_type, device=device, n_few_shots=n_few_shots, thinking=thinking)
    if task == "mmar":
        return MMAREvaluator(model, dataset, model_type=model_type, device=device, n_few_shots=n_few_shots, thinking=thinking)
    if task.startswith("multi_challenge"):
        return MultiChallengeEvaluator(model, dataset, task, model_type=model_type, device=device, n_few_shots=n_few_shots, thinking=thinking)
    if task.startswith("bigbench_audio"):
        return BigBenchAudioEvaluator(model, dataset, task, model_type=model_type, device=device)
    if task == "mmau_pro":
        return MMAUProEvaluator(model, dataset, model_type=model_type, device=device, n_few_shots=n_few_shots, thinking=thinking)