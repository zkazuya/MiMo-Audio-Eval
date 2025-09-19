# Copyright 2025 Xiaomi Corporation.
def get_model(model, model_type=None, model_path=None, tokenizer_path=None, device=None):
    if model == "baichuan_audio":
        from .baichuan_audio import BaichuanAudio
        return BaichuanAudio(model_type)
    if model == "kimi_audio":
        from .kimi_audio import KimiAudioModel
        return KimiAudioModel(model_type)
    if model == "qwen_omni":
        from .qwen_omni import QwenOmniModel
        return QwenOmniModel(device)
    if model == "step_audio2":
        from .step_audio2 import StepAudio2Model
        return StepAudio2Model(model_type)
    if model == "mimo_audio":
        from .mimo_audio import MimoAudioModel
        return MimoAudioModel(model_path, tokenizer_path)