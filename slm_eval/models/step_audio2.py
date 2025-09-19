# Copyright 2025 Xiaomi Corporation.
import os
import re
import sys
import copy
import torch
import tempfile
import numpy as np
import soundfile as sf 
import s3tokenizer
from typing import List, Union
from pathlib import Path
from .src_step_audio.stepaudio2 import StepAudio2, StepAudio2Base
from .src_step_audio.token2wav import Token2wav

sys.path.append(Path(__file__).parent / "src_step_audio")
PROMPT_WAV=Path(__file__).parent / "src_step_audio/assets/default_female.wav"

StepAudio_MODEL_PATH = {
    "base": "stepfun-ai/Step-Audio-2-mini-Base",
    "instruct": "stepfun-ai/Step-Audio-2-mini",
}

audio_sampling_params = {
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "do_sample": True
}

text_sampling_params = {
    "max_tokens": 2048,
    "temperature": 0.5,
    "max_new_tokens": 2048,
    "temperature": 0.1,
    "do_sample": True
}

class StepAudio2Model:

    def __init__(self, model_type):
        self.model = self.load_model(model_type)
        self.token_wav = self.load_wav(model_type)
        self.history = []

    def load_wav(self, model_type):
        model_path = '/'.join([StepAudio_MODEL_PATH[model_type], 'token2wav'])
        print(f"Loading model from {model_path} ...")

        return Token2wav(model_path)

    def load_model(self, model_type):
        model_path = StepAudio_MODEL_PATH[model_type]
        print(f"Loading model from {model_path} ...")
        model_class = StepAudio2 if model_type == "instruct" else StepAudio2Base
        model = model_class(model_path=model_path)
        return model
    
    def audio2token(self, audio):
        audio_tokenizer = self.token_wav.audio_tokenizer
        audio = s3tokenizer.load_audio(audio, sr=16000)
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        speech_tokens, _ = audio_tokenizer.quantize(mels.cuda(), mels_lens.cuda())
        return speech_tokens

    @torch.no_grad()
    def convert_instructions(self, instructions, td):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        sound = "<sound>"
        audio_wavs_index = 0
        for ins in instructions:
            if ins["from"] == "human":
                text_information = ins["value"][0]["value"]
                content_list = []
                if sound in text_information:
                    print(text_information)
                    left_text, right_text = text_information.split(sound)
                    if left_text != "":
                        content_list.append({"type": "text", "text": left_text})
                    if isinstance(ins["value"][1]["value"], str):
                        content_list.append({"type": "audio", "audio": ins["value"][1]["value"]})
                    else:
                        audio_path = os.path.join(td, str(audio_wavs_index) + '.wav')
                        sf.write(audio_path, ins["value"][1]["value"].cpu().view(-1).numpy(), 24000)
                        audio_wavs_index += 1
                        content_list.append({"type": "audio", "audio": audio_path})
                    if right_text != "":
                        content_list.append({"type": "text", "text": right_text})
                else:
                    content_list.append({"type": "text", "text": text_information})
                messages.append({"role": "human", "content": content_list})
                messages.append({"role": "assistant", "content": None})
        return messages
    
    @torch.no_grad()
    def instruction_following(self, instructions, append_generation_prompt=False, thinking=False):
        with tempfile.TemporaryDirectory() as td:
            messages = self.convert_instructions(instructions, td)
            tokens, text, _ = self.model(messages, **text_sampling_params)
            return text

    @torch.no_grad()
    def instruction_following_for_step2_audio_mmau(self, instructions, append_generation_prompt=False, thinking=False):
        with tempfile.TemporaryDirectory() as td:
            messages = [{"role": "system", "content": "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately."}]
            sound = "<sound>"
            audio_wavs_index = 0
            if isinstance(instructions[0]["value"][1]["value"], str):
                audio_path = instructions[0]["value"][1]["value"]
            else:
                audio_path = os.path.join(td, str(audio_wavs_index) + '.wav')
                sf.write(audio_path, instructions[0]["value"][1]["value"].cpu().view(-1).numpy(), 24000)
            text_information = instructions[0]["value"][0]["value"].split(sound)[-1]
            messages.append({"role": "human", "content": [{"type": "text", "text": text_information}, {"type": "audio", "audio": audio_path}]},)
            messages.append({"role": "assistant", "content": None})
            print(messages)
            tokens, text, _ = self.model(messages, max_new_tokens=256, num_beams=1)
            print(text)
            return text

    @torch.no_grad()
    def speech2text_dialogue_sft(self, audio, add_history=False):
        if add_history == False:
            self.history = [{"role": "system", "content": "You are a helpful assistant."}]

        import copy
        messages = copy.deepcopy(self.history)

        with tempfile.TemporaryDirectory() as td:
            if isinstance(audio, str) == False:
                input_path = os.path.join(td, "input.wav")
                sf.write(input_path, audio.cpu().view(-1).numpy(), 24000)
            else:
                input_path = audio
            messages.append({"role": "human", "content": [{"type": "audio", "audio": input_path}]})
            messages.append({"role": "assistant", "content": None})
            tokens, text, _ = self.model(messages, **text_sampling_params)

            print(text)
            self.history.append({"role": "assistant", "content": text})
            return text

    @torch.no_grad()
    def spoken_dialogue_sft_multiturn(self, input_speech, text_dialogue, output_audio_path=None, prompt_speech=None):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        
        for i in range(len(input_speech)):
            if input_speech[i]['role'] == 'user':
                messages.append({"role": "human", "content": [{"type": "audio", "audio": input_speech[i]['content']}]})
            else:
                assist_audio_path = input_speech[i]['content']
                audio = s3tokenizer.load_audio(assist_audio_path, sr=16000)  # [T]
                mels = s3tokenizer.log_mel_spectrogram(audio)
                mels, mels_lens = s3tokenizer.padding([mels])
                speech_tokens, _ = self.token_wav.audio_tokenizer.quantize(mels.cuda(), mels_lens.cuda())
                messages.append({"role": "assistant", "content":[{"type": "text", "text":"<tts_start>"}, {"type":"token", "token": speech_tokens.cpu().tolist()[0]}]})
        
        messages.append({"role": "assistant", "content": "<tts_start>", "eot": False})
        tokens, text, audio = self.model(messages, **audio_sampling_params)
        print(text)
        audio = [x for x in audio if x < 6561]
        audio = self.token_wav(audio, prompt_wav=prompt_speech)

        if output_audio_path is not None:
            with open(output_audio_path, 'wb') as f:
                f.write(audio)

        return text

    @torch.no_grad()
    def audio_understanding_sft(self, audio, input_text, thinking=False):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.append({"role": "human", "content": [{"type": "text", "text": input_text}, {"type": "audio", "audio": audio}]})
        messages.append({"role": "assistant", "content": None})
        tokens, text, _ = self.model(messages, **text_sampling_params)
        return text

    @torch.no_grad()
    def asr_sft(self, audio, lang='zh'):
        messages = [{"role": "system", "content": "请记录下你所听到的语音内容。"}]
        messages.append({"role": "human", "content": [{"type": "audio", "audio": audio}]})
        messages.append({"role": "assistant", "content": None})
        tokens, text, _ = self.model(messages, max_new_tokens=256)
        return text.strip("<中文>").strip("<英语>")

    def detect_language(self, text):
        chinese = re.findall(r'[\u4e00-\u9fff]', text)
        return "zh" if chinese else "en"

    def tts_sft(self, text, output_path, instruction=None):
        language = self.detect_language(text)
        if language == "en":
            messages = [{"role": "system", "content": "Please synthesize the given text into speech."}]
        else:
            messages = [{"role": "system", "content": "请将以下文本合成为自然流畅的语音。"}]
        messages.append({"role": "human", "content": [{"type": "text", "text": text}]})
        messages.append({"role": "assistant", "content": "<tts_start>", "eot": False})

        tokens, text, audio = self.model(messages, **audio_sampling_params)

        audio = [x for x in audio if x < 6561]
        audio = self.token_wav(audio, prompt_wav=PROMPT_WAV)
        if output_path is not None:
            with open(output_path, 'wb') as f:
                f.write(audio)
        return text

    @torch.no_grad()
    def spoken_dialogue_sft(self, audio, output_audio_path=None, add_history=False, prompt_speech=None):
        if add_history == False:
            self.history = [{"role": "system", "content": "You are a helpful assistant."}]

        messages = copy.deepcopy(self.history)

        with tempfile.TemporaryDirectory() as td:
            if isinstance(audio, str):
                input_path = audio
            else:
                input_path = os.path.join(td, "input.wav")
                sf.write(input_path, audio.cpu().view(-1).numpy(), 24000)  # 写盘
            messages.append({"role": "human", "content": [{"type": "audio", "audio": input_path}]})
            messages.append({"role": "assistant", "content": "<tts_start>", "eot": False})
            tokens, text, audio = self.model(messages, **audio_sampling_params)

            print(text)

            audio = [x for x in audio if x < 6561] # remove audio padding
            audio = self.token_wav(audio, prompt_wav=PROMPT_WAV)
            
            if output_audio_path is not None:
                with open(output_audio_path, 'wb') as f:
                    f.write(audio)

            self.history.append({"role": "human", "content": [{"type": "audio", "audio": input_path}]})
            self.history.append(
                {
                    "role": "assistant",
                    "content":[
                        {"type": "text", "text":"<tts_start>"},
                        {"type":"token", "token": tokens}
                    ]
                }
            )
            return text
    
    @torch.no_grad()
    def gen_text(self, prompts):
        messages = []
        for prompt in prompts:
            if prompt[1] == "speech":
                messages.append({"type": "audio", "audio": str(prompt[0]["speech"])})
            else:
                messages.append(prompt[0]["text"])
        text_sampling_params["do_sample"] = False
        text_sampling_params["max_new_tokens"] = 10
        tokens, text, _ = self.model(messages, **text_sampling_params)
        return text
    
    @torch.no_grad()
    def gen_speech(self, prompts, output_audio_path):
        messages = []
        for prompt in prompts:
            if prompt[1] == "speech":
                messages.append({"type": "audio", "audio": str(prompt[0]["speech"])})
            else:
                messages.append(prompt[0]["text"])
                    
        messages.append("<tts_start>")
        audio_sampling_params["max_new_tokens"] = 100
        tokens, text, audio = self.model(messages, **audio_sampling_params)
        audio = [x for x in audio if x < 6561]
        if len(audio) == 0:
            audio = torch.zeros(24000)
            sf.write(output_audio_path, audio.cpu().view(-1).numpy(), 24000)
            return text
        audio = self.token_wav(audio, prompt_wav=PROMPT_WAV)
        with open(output_audio_path, 'wb') as f:
            f.write(audio)
        return text
    
    @torch.no_grad()
    def few_shots_qa(self, examples):
        messages = []
        for i in range(len(examples[:-1])):
            messages.append({"type": "audio", "audio": str(examples[i]["audio_path"])})
            messages.append("\nQuestion: " + examples[i]["question"] + " \n\n")
            messages.append("Answer: " + examples[i]["answer"] + " \n\n")
        
        messages.append({"type": "audio", "audio": str(examples[-1]["audio_path"])})
        messages.append("\nQuestion: " + examples[-1]["question"] + " \n\n")
        messages.append("Answer: ")
        text_sampling_params["do_sample"] = False
        text_sampling_params["max_new_tokens"] = 100
        tokens, text, _ = self.model(messages, **text_sampling_params)
        return text