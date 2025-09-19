# Copyright 2025 Xiaomi Corporation.
import torch
import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


class QwenOmniModel:

    def __init__(self, device):
        self.model, self.processor = self.load_model(device)

    def load_model(self, device):
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B", 
            torch_dtype="auto",
            attn_implementation="flash_attention_2"
        ).to(device)
        processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        return model, processor

    @torch.no_grad()
    def gen_text(self, prompts):
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
        ]
        for prompt in prompts:
            role = "user" if prompt[2] == "input" else "assistant"
            if prompt[1] == "speech":
                content = [{"type": "audio", "audio": str(prompt[0]["speech"])}]
            else:
                content = [{"type": "text", "text": prompt[0]["text"]}]
            conversation.append({
                "role": role,
                "content": content
            })
            
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        text_ids = self.model.generate(**inputs, thinker_max_new_tokens=10, use_audio_in_video=False, return_audio=False)
        text = self.processor.batch_decode(text_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text
    
    @torch.no_grad()
    def gen_speech(self, prompts, output_audio_path):
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
        ]
        for prompt in prompts:
            role = "user" if prompt[2] == "input" else "assistant"
            if prompt[1] == "speech" and role == "user":
                content = [{"type": "audio", "audio": str(prompt[0]["speech"])}]
            else:
                content = [{"type": "text", "text": prompt[0]["text"]}]
            conversation.append({
                "role": role,
                "content": content
            })
            
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        text_ids, audio = self.model.generate(**inputs, thinker_max_new_tokens=10, use_audio_in_video=False)
        sf.write(
            output_audio_path,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        text = self.processor.batch_decode(text_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text

    def detect_language(self, text):
        import re
        chinese = re.findall(r'[\u4e00-\u9fff]', text)
        
        if chinese:
            return "zh"
        else:
            return "en"

    @torch.no_grad()
    def asr_sft(self, audio_path, lang="zh"):
        if lang == "zh":
            instruction = "请把这段语音转录成文本，不要有任何解释。"
        else:
            instruction = "Please transcribe the spoken content into written text, without any explanation."
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": instruction}, {"type": "audio", "audio": audio_path}]
            },
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        text_ids, audio = self.model.generate(**inputs, use_audio_in_video=False)
        text = self.processor.batch_decode(text_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text

    @torch.no_grad()
    def tts_sft(self, text, output_path):
        if self.detect_language(text) == "zh":
            instruction = "请把以下的文本转换成语音，不要有任何解释：\n\n"
        else:
            instruction = "Please convert the following text to speech, without any explanation: \n\n"
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": instruction + text}]
            }
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        text_ids, audio = self.model.generate(**inputs, use_audio_in_video=False)
        sf.write(output_path, audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
        text = self.processor.batch_decode(text_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text

    @torch.no_grad()
    def audio_understanding_sft(self, audio_path, input_text, thinking=False):
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
            },
            {
                "role": "user",
                "content": [ {"type": "text", "text": input_text}, {"type": "audio", "audio": audio_path}]
            },
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        text_ids, audio = self.model.generate(**inputs, use_audio_in_video=False)
        text = self.processor.batch_decode(text_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text

    @torch.no_grad()
    def spoken_dialogue_sft_multiturn(self, input_speech, text_dialogue, output_audio_path=None, prompt_speech=None):
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
        ]
        for i, prompt in enumerate(input_speech):
            role = "user" if input_speech[i]['role'] == 'user' else "assistant"
            content = [{"type": "audio", "audio": input_speech[i]["content"]}]
            conversation.append({
                "role": role,
                "content": content
            })

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        text_ids, audio = self.model.generate(**inputs, use_audio_in_video=False)
        sf.write(
            output_audio_path,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        text = self.processor.batch_decode(text_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return text

    @torch.no_grad()
    def speech2text_dialogue_sft(self, audio):
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
        ]
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.wav")
            sf.write(input_path, audio.cpu().view(-1).numpy(), 24000)  # 写盘
            conversation.append({
                "role": 'user',
                "content": [{"type": "audio", "audio": input_path}]
            })
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            text_ids, _ = self.model.generate(**inputs, use_audio_in_video=False)
            text = self.processor.batch_decode(text_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            return text

    @torch.no_grad()
    def spoken_dialogue_sft(self, audio, output_audio_path=None, add_history=False, prompt_speech=None):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
        ]
        if add_history:
            import copy
            messages = copy.deepcopy(self.history)
        else:
            self.history = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },
            ]

        import tempfile
        import os
        with tempfile.TemporaryDirectory() as td:
            if isinstance(audio, str):
                input_path = audio
            else:
                input_path = os.path.join(td, "input.wav")
                sf.write(input_path, audio.cpu().view(-1).numpy(), 24000)  # 写盘
            messages.append({
                "role": 'user',
                "content": [{"type": "audio", "audio": input_path}]
            })

            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            text_ids, audio = self.model.generate(**inputs, use_audio_in_video=False)
            sf.write(
                output_audio_path,
                audio.reshape(-1).detach().cpu().numpy(),
                samplerate=24000,
            )
            text = self.processor.batch_decode(text_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


            self.history.append({
                "role": 'user',
                "content": [{"type": "audio", "audio": input_path}]
            })
            self.history.append({
                "role": "assistant",
                "content": [{"type": "audio", "audio": output_audio_path}]
            })

            return text

    @torch.no_grad()
    def convert_instructions(self, instructions, td):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
        ]
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
                        import os
                        audio_path = os.path.join(td, str(audio_wavs_index) + '.wav')
                        sf.write(audio_path, ins["value"][1]["value"].cpu().view(-1).numpy(), 24000)
                        audio_wavs_index += 1
                        content_list.append({"type": "audio", "audio": audio_path})
                    if right_text != "":
                        content_list.append({"type": "text", "text": right_text})
                else:
                    content_list.append({"type": "text", "text": text_information})
                messages.append({
                    "role": "user",
                    "content": content_list
                })
            else:
                text_information = ins["value"][0]["value"]
                content_list = []
                if sound in text_information:
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
                messages.append({
                    "role": "assistant",
                    "content": content_list
                })
        return messages
    
    @torch.no_grad()
    def instruction_following(self, instructions, append_generation_prompt=False, thinking=False):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            messages = self.convert_instructions(instructions, td)
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            text_ids, _ = self.model.generate(**inputs, use_audio_in_video=False)
            text = self.processor.batch_decode(text_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            return text

    @torch.no_grad()
    def qa(self, input_speech, input_text):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.wav")
            sf.write(input_path, input_speech.cpu().view(-1).numpy(), 16000)  # 写盘

            content_list = []
            if input_text != "":
                content_list.append({"type": "text", "text": input_text})
            content_list.append({"type": "audio", "audio": input_path})

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },
            ]
            messages.append({
                "role": "user",
                "content": content_list
            })
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            text_ids, _ = self.model.generate(**inputs, use_audio_in_video=False)
            text = self.processor.batch_decode(text_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            return text
