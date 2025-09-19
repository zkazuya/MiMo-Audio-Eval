# Copyright 2025 Xiaomi Corporation.
import os
import copy
import torch
import tempfile
import numpy as np
import soundfile as sf 
from typing import List, Union
from torch.nn import CrossEntropyLoss
from slm_eval.models.src_kimi_audio.kimia_infer.api.kimia import KimiAudio

KimiAudio_MODEL_PATH = {
    "base": "moonshotai/Kimi-Audio-7B",
    "instruct": "moonshotai/Kimi-Audio-7B-Instruct"
}

sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

class KimiAudioModel:

    def __init__(self, model_type):
        self.model = self.load_model(model_type)
        self.history = []

    def load_model(self, model_type):
        if 'instruct' in model_type:
            model_path = KimiAudio_MODEL_PATH["instruct"]
        else:
            model_path = KimiAudio_MODEL_PATH['base']
        print(f"Loading model from {model_path} ...")
        model = KimiAudio(model_path=model_path, load_detokenizer=True)
        return model

    @torch.no_grad()
    def few_shots_qa(self, examples, max_new_tokens=1024):
        
        messages = []

        for i in range(len(examples[:-1])):
            messages.append({"role": "user", "message_type": "text", "content": "\nQuestion: " + examples[i]["question"] + " \n\n"})
            messages.append({"role": "user", "message_type": "audio", "content": examples[i]["audio_path"]})
            messages.append({"role": "user", "message_type": "text", "content": "Answer: " + examples[i]["answer"] + " \n\n"})
        
        messages.append({"role": "user", "message_type": "text", "content": "Question: " + examples[-1]["question"] + " \n\n"})
        messages.append({"role": "user", "message_type": "audio", "content": examples[-1]["audio_path"]})
        messages.append({"role": "user", "message_type": "text", "content": "Answer: "})
        
        _, text_output = self.model.generate_in_context_learning(messages, max_new_tokens=max_new_tokens, output_type="text")
        return text_output
    
    @torch.no_grad()
    def spoken_dialogue_sft_multiturn(self, input_speech, text_dialogue, output_audio_path=None, prompt_speech=None):
        messages = []
        for i in range(len(input_speech)):
            if input_speech[i]['role'] == 'user':
                messages.append({"role": "user", "message_type": "audio", "content": input_speech[i]['content']})
            else:
                messages.append({"role": "assistant", "message_type": "audio", "content": input_speech[i]['content']})

        wav_output, text_output = self.model.generate(messages, **sampling_params, output_type="both")
        if output_audio_path is not None:
            sf.write(output_audio_path, wav_output.detach().cpu().view(-1).numpy(), 24000)

        return text_output

    @torch.no_grad()
    def audio_understanding_sft(self, audio, input_text, thinking=False):
        messages = [{"role": "user", "message_type": "text", "content": input_text}]
        messages.append({"role": "user", "message_type": "audio", "content": audio})
        _, text_output = self.model.generate(messages, **sampling_params, output_type="text")
        return text_output
    
    @torch.no_grad()
    def asr(self, audio):
        messages = [
            {"role": "user", "message_type": "text", "content": "Please transcribe the following audio:"},
            {"role": "user", "message_type": "audio", "content": audio}
        ]
        _, text_output = self.model.generate(messages, **sampling_params, output_type="text")
        return text_output
    
    @torch.no_grad()
    def asr_sft(self, audio, lang='zh'):
        if lang == 'zh':
            instruction = "请把这段语音转录成文本。"
        else:
            instruction = "Please transcribe the spoken content into written text."
        messages = [
            {"role": "user", "message_type": "text", "content": instruction},
            {"role": "user", "message_type": "audio", "content": audio}
        ]
        _, text_output = self.model.generate(messages, **sampling_params, output_type="text")
        return text_output


    def detect_language(self, text):
        # 中文的 Unicode 范围 \u4e00-\u9fff
        import re
        chinese = re.findall(r'[\u4e00-\u9fff]', text)
        
        if chinese:
            return "zh"
        else:
            return "en"

    @torch.no_grad()
    def tts_sft(self, ref, output_path):
        if self.detect_language(ref) == 'zh':
            instruction = "请把以下的文本转换成语音。"
        else:
            instruction = "Please convert the following text to speech."
        messages = [
            {"role": "user", "message_type": "text", "content": instruction + " \n\n" + ref},
            # {"role": "user", "message_type": "text", "content": ref}
        ]

        wav_output, text_output = self.model.generate(messages, **sampling_params, output_type="both")
        if output_path is not None:
            sf.write(output_path, wav_output.detach().cpu().view(-1).numpy(), 24000)
        return text_output


    @torch.no_grad()
    def convert_instructions(self, instructions, td):
        messages = []
        sound = "<sound>"
        audio_wavs_index = 0
        for ins in instructions:
            if ins["from"] == "human":
                text_information = ins["value"][0]["value"]
                if sound in text_information:
                    print(text_information)
                    left_text, right_text = text_information.split(sound)
                    if left_text != "":
                        messages.append({"role": "user", "message_type": "text", "content": left_text})
                    if isinstance(ins["value"][1]["value"], str):
                        messages.append({"role": "user", "message_type": "audio", "content": ins["value"][1]["value"]})
                    else:
                        audio_path = os.path.join(td, str(audio_wavs_index) + '.wav')
                        sf.write(audio_path, ins["value"][1]["value"].cpu().view(-1).numpy(), 24000)
                        audio_wavs_index += 1
                        messages.append({"role": "user", "message_type": "audio", "content": audio_path})
                    if right_text != "":
                        messages.append({"role": "user", "message_type": "text", "content": right_text})
                else:
                    messages.append({"role": "user", "message_type": "text", "content": text_information})
            else:
                text_information = ins["value"][0]["value"]
                if sound in text_information:
                    left_text, right_text = text_information.split(sound)
                    if left_text != "":
                        messages.append({"role": "assistant", "message_type": "text", "content": left_text})
                    if isinstance(ins["value"][1]["value"], str):
                        messages.append({"role": "user", "message_type": "audio", "content": ins["value"][1]["value"]})
                    else:
                        audio_path = os.path.join(td, str(audio_wavs_index) + '.wav')
                        sf.write(audio_path, ins["value"][1]["value"].cpu().view(-1).numpy(), 24000)
                        audio_wavs_index += 1
                        messages.append({"role": "user", "message_type": "audio", "content": audio_path})
                    if right_text != "":
                        messages.append({"role": "assistant", "message_type": "text", "content": right_text})
                else:
                    messages.append({"role": "assistant", "message_type": "text", "content": text_information})
        return messages
    
    @torch.no_grad()
    def instruction_following(self, instructions, append_generation_prompt=False, thinking=False):
        with tempfile.TemporaryDirectory() as td:
            messages = self.convert_instructions(instructions, td)
            _, text_output = self.model.generate(messages, **sampling_params, output_type="text")
            return text_output
    
    @torch.no_grad()
    def speech2text_dialogue_sft(self, audio):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.wav")
            sf.write(input_path, audio.cpu().view(-1).numpy(), 24000)  # 写盘
            messages = [{"role": "user", "message_type": "audio", "content": input_path}]
            _, text_output = self.model.generate(messages, **sampling_params, output_type="text")
            return text_output

    @torch.no_grad()
    def spoken_dialogue_sft(self, audio, output_audio_path=None, add_history=False, prompt_speech=None):
        messages = []
        if add_history:
            import copy
            messages = copy.deepcopy(self.history)
        else:
            self.history = []

        with tempfile.TemporaryDirectory() as td:
            if isinstance(audio, str):
                input_path = audio
            else:
                input_path = os.path.join(td, "input.wav")
                sf.write(input_path, audio.cpu().view(-1).numpy(), 24000)  # 写盘
            messages.append({"role": "user", "message_type": "audio", "content": input_path})
            wav_output, text_output = self.model.generate(messages, **sampling_params, output_type="both")
            if output_audio_path is not None:
                sf.write(output_audio_path, wav_output.detach().cpu().view(-1).numpy(), 24000)
                
            self.history.append({"role": "user", "message_type": "audio", "content": input_path})
            self.history.append({"role": "assistant", "message_type": "audio", "content": output_audio_path})
            return text_output

    @torch.no_grad()
    def gen_text(self, prompts):
        messages = []
        for prompt in prompts:
            role = "user" if prompt[2] == "input" else "assistant"
            if prompt[1] == "speech":
                messages.append({"role": role, "message_type": "audio", "content": str(prompt[0]["speech"])})
            else:
                messages.append({"role": role, "message_type": "text", "content": prompt[0]["text"]})
        _, text_output = self.model.generate(messages, **sampling_params, max_new_tokens=5, output_type="text")
        return text_output
    
    @torch.no_grad()
    def qa(self, input_speech, input_text):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.wav")
            sf.write(input_path, input_speech.cpu().view(-1).numpy(), 16000)  # 写盘
            
            messages = [{"role": "user", "message_type": "audio", "content": input_path}]
            if input_text != "":
                messages.append({"role": "user", "message_type": "text", "content": input_text})
            _, text_output = self.model.generate(messages, **sampling_params, max_new_tokens=1024, output_type="text")
            return text_output

    @torch.no_grad()
    def gen_speech(self, prompts, output_audio_path):
        messages = []
        for prompt in prompts:
            role = "user" if prompt[2] == "input" else "assistant"
            if role == "user":
                if prompt[1] == "speech":
                    messages.append({"role": role, "message_type": "audio", "content": str(prompt[0]["speech"])})
                else:
                    messages.append({"role": role, "message_type": "text", "content": prompt[0]["text"]})
            else:
                messages.append({"role": role, "message_type": "both", "content": prompt[0]})
        
        output_type = "both"
        msgs = []
        tokenize_role = True
        has_ct_token = False
        has_msg_end_token = False
        previous_role = None
        for msg_idx, message in enumerate(messages):
            if previous_role is None:
                tokenize_role = True
            else:
                if message["role"] == previous_role:
                    tokenize_role = False
                else:
                    tokenize_role = True
            
            if msg_idx == len(messages) - 1:
                has_ct_token = True
                has_msg_end_token = True
            else:
                if messages[msg_idx + 1]["role"] != message["role"]:
                    has_ct_token = True
                    has_msg_end_token = True
                else:
                    has_ct_token = False
                    has_msg_end_token = False

            previous_role = message["role"]

            if message["role"] == "user":
                msg = self.model.prompt_manager.tokenize_message(
                    message=message,
                    tokenize_role=tokenize_role,
                    has_ct_token=has_ct_token,
                    has_msg_end_token=has_msg_end_token,
                    extract_whisper_feature=True,
                    output_type=output_type,
                )
            else:
                msg_speech = self.model.prompt_manager.tokenize_message(
                    message={"role": "assistant", "message_type": "audio", "content": message["content"]["speech"]},
                    tokenize_role=tokenize_role,
                    has_ct_token=has_ct_token,
                    has_msg_end_token=has_msg_end_token,
                    extract_whisper_feature=False,
                    output_type=output_type,
                )
                msg_text = self.model.prompt_manager.tokenize_message(
                    message={"role": "assistant", "message_type": "text", "content": message["content"]["text"]},
                    tokenize_role=tokenize_role,
                    has_ct_token=has_ct_token,
                    has_msg_end_token=has_msg_end_token,
                    extract_whisper_feature=False,
                    output_type=output_type,
                )
                msg = msg_speech
                msg.audio_token_ids = [msg_speech.audio_token_ids[0]] + [self.model.prompt_manager.extra_tokens.kimia_text_blank] * 6 + msg_speech.audio_token_ids[2:-3] + msg_speech.audio_token_ids[-2:]
                msg.text_token_ids = [self.model.prompt_manager.extra_tokens.kimia_text_blank] * len(msg.audio_token_ids)
                msg.text_token_ids[1:len(msg_text.text_token_ids)-1] = msg_text.text_token_ids[1:-1]
                msg.is_continuous_mask = [False] * len(msg.audio_token_ids)

            msgs.append(msg)

        assistant_start_msg = self.model.prompt_manager.tokenize_message(
            message={
                "role": "assistant",
                "message_type": None,
            },
            tokenize_role=True,
            has_ct_token=False,
            has_msg_end_token=False,
        )
        msgs.append(assistant_start_msg)

        ret_msg = msgs[0]
        for msg in msgs[1:]:
            ret_msg.merge(msg)

        wav_output, text_output = self.model.generate(chats=None, history=ret_msg, **sampling_params, max_new_tokens=30, output_type="both")
        sf.write(output_audio_path, wav_output.detach().cpu().view(-1).numpy(), 24000)

        return text_output