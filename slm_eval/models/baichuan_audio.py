# Copyright 2025 Xiaomi Corporation.
import os
import re
import sys
import json
import torch
import tempfile
import numpy as np
import soundfile as sf 
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from slm_eval.models.src_baichuan_audio.generation import GenerationAudioTokens, decode_save_concat
from typing import List


BaichuanAudio_MODEL_PATH = {
    "base": "baichuan-inc/Baichuan-Audio-Base",
    "instruct": "baichuan-inc/Baichuan-Audio-Instruct"
}
COSY_VOCODER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src_baichuan_audio/third_party/cosy24k_vocoder")

sampling_rate = 24000
wave_concat_overlap = int(sampling_rate * 0.01)


class BaichuanAudio:

    def __init__(self, model_type):
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_type)
        self.model.bind_processor(self.tokenizer, training=False, relative_path='/')
        self.vocoder = self.load_vocoder()
    
    def load_model_and_tokenizer(self, model_type):
        if 'instruct' in model_type:
            model_path = BaichuanAudio_MODEL_PATH["instruct"]
        else:
            model_path = BaichuanAudio_MODEL_PATH['base']
        print(f"Loading model from {model_path} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
            model_max_length=128000,
        )
        device_map = 'cuda'
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        model.config.use_cache = True
        return model, tokenizer
    
    def load_vocoder(args):
        sys.path.append(os.path.join(COSY_VOCODER))
        from cosy24k_vocoder import Cosy24kVocoder
        vocoder = Cosy24kVocoder.from_pretrained(os.path.join(COSY_VOCODER, "hift.pt"))
        vocoder = vocoder.cuda()
        return vocoder
    
    def asr(self, audio):
        audio_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_start_token_id)
        audio_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_end_token_id)
        PROMPT_ASR = '将语音转录为文本:' + audio_start_token + '{}' + audio_end_token
        prompt = PROMPT_ASR.format(json.dumps({'path': audio}))
        ret = self.model.processor([prompt])
        predicted_ids = self.model.generate(
            input_ids=ret.input_ids.cuda(),
            attention_mask=ret.attention_mask.cuda(),
            labels=None,
            audios=ret.audios.cuda() if ret.audios is not None else None,
            encoder_length=ret.encoder_length.cuda() if ret.encoder_length is not None else None,
            bridge_length=ret.bridge_length.cuda() if ret.bridge_length is not None else None,
            max_new_tokens=700,
            num_beams=1,
            do_sample=False, 
            top_k=5, 
            top_p=0.85, 
            temperature=0.5,
            num_return_sequences=1,
            repetition_penalty=1.3,
        )
        generated = self.tokenizer.batch_decode(predicted_ids[:,ret.input_ids.shape[1]:], skip_special_tokens=True)
        return generated[0]
    
    def split_string_with_punctuation_merged(self, s):
        pattern = r'([:,;!?，。；：！？])'
        punctuation_positions = [(m.start(0), m.group(0)) for m in re.finditer(pattern, s)]
        substrings = []
        last_index = 0
        for pos, punct in punctuation_positions:
            substrings.append(s[last_index:pos] + punct)
            last_index = pos + len(punct)
        if last_index < len(s):
            substrings.append(s[last_index:])
        return substrings

    @torch.no_grad()
    def tts(self, text, output_path):
        PROMPT_TTS='{}' + self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audiogen_start_token_id)
        prompts = []
        cur_prompt = ""
        for t in self.split_string_with_punctuation_merged(text):
            cur_prompt += t
            if len(cur_prompt) > 10:
                prompts.append(PROMPT_TTS.format(cur_prompt))
                cur_prompt = ""
        if cur_prompt:
            prompts.append(PROMPT_TTS.format(cur_prompt))
        audio_response = []
        gret = None
        for prompt in prompts:
            ret = self.model.processor([prompt])
            if gret is not None:
                gret.sequences[:,-1] = self.model.config.audio_config.audiogen_end_token_id
                gret.sequences = torch.concat([gret.sequences, ret.input_ids.cuda()], dim=1)
            gret = GenerationAudioTokens.generate(
                self.model, 
                input_ids=ret.input_ids.cuda() if gret is None else gret.sequences.cuda(),
                labels=None,
                audios=ret.audios.cuda() if ret.audios is not None else None,
                encoder_length=ret.encoder_length.cuda() if ret.encoder_length is not None else None,
                bridge_length=ret.bridge_length.cuda() if ret.bridge_length is not None else None,
                past_key_values= gret.past_key_values if gret is not None else None,
                max_new_tokens=700,
                num_beams=1,
                do_sample=False, 
                top_k=5, 
                top_p=0.85, 
                temperature=0.5,
                num_return_sequences=1,
                repetition_penalty=1.3,
                return_dict_in_generate=True,
            )
            audio_response.append(gret.audios_sequences)
        decode_save_concat(audio_response, self.vocoder, self.model, str(output_path), sampling_rate, wave_concat_overlap)
    
    def tokenize_audio(self, audio_paths):
        audio_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_start_token_id)
        audio_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_end_token_id)
        audio_tokens_str_list = [audio_start_token + json.dumps({'path': str(audio_path)}) + audio_end_token for audio_path in audio_paths]
        return audio_tokens_str_list
    
    def tokenize_audio_gen(self, audio_paths):
        audiogen_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audiogen_start_token_id)
        audiogen_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audiogen_end_token_id)
        audio_tokens_str_list = [audiogen_start_token + json.dumps({'path': str(audio_path)}) + audiogen_end_token for audio_path in audio_paths]
        return audio_tokens_str_list
        
    def few_shots_qa(self, examples, max_new_tokens=1024):
        audio_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_start_token_id)
        audio_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_end_token_id)
        final_input = ''
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(len(examples[:-1])):
                final_input += 'Question: ' + examples[i]["question"] + ' \n\n'
                audio_path = examples[i]["audio_path"]
                import torchaudio
                read_audio_lines, sample_rate = torchaudio.load(audio_path)
                channel_number = read_audio_lines.size(0)
                if channel_number > 1:
                    file_path = os.path.join(tmpdir, f"{i}.wav")
                    import soundfile as sf
                    sf.write(file_path, read_audio_lines[0].cpu().numpy(), sample_rate)
                else:
                    file_path = audio_path
                final_input += audio_start_token + json.dumps({'path': file_path}) + audio_end_token
                final_input += '\nAnswer: ' + examples[i]["answer"] + ' \n\n'
            final_input += '\nQuestion: ' + examples[-1]["question"] + ' \n\n'
            
            import torchaudio
            read_audio_lines, sample_rate = torchaudio.load(examples[-1]["audio_path"])
            channel_number = read_audio_lines.size(0)
            if channel_number > 1:
                file_path = os.path.join(tmpdir, f"input.wav")
                import soundfile as sf
                sf.write(file_path, read_audio_lines[0].cpu().numpy(), sample_rate)
            else:
                file_path = audio_path
            
            final_input += audio_start_token + json.dumps({'path': file_path}) + audio_end_token + '\n' 
            final_input += 'Answer: '

            ret = self.model.processor([final_input])
            predicted_ids = self.model.generate(
                input_ids=ret.input_ids.cuda(),
                attention_mask=ret.attention_mask.cuda(),
                labels=None,
                audios=ret.audios.cuda() if ret.audios is not None else None,
                encoder_length=ret.encoder_length.cuda() if ret.encoder_length is not None else None,
                bridge_length=ret.bridge_length.cuda() if ret.bridge_length is not None else None,
                max_new_tokens=700,
                num_beams=1,
                do_sample=False, 
                top_k=5, 
                top_p=0.85, 
                temperature=0.5,
                num_return_sequences=1,
                repetition_penalty=1.3,
            )
            generated = self.tokenizer.batch_decode(predicted_ids[:,ret.input_ids.shape[1]:], skip_special_tokens=True)
            return generated[0]

    @torch.no_grad()
    def gen_text(self, prompts):
        audio_tokens_str_list = [
            self.tokenize_audio([prompt[0]["speech"]])[0] if prompt[2] == "input" else self.tokenize_audio_gen([prompt[0]["speech"]])[0]
            for prompt in prompts if prompt[1] == "speech"
        ]
        full_prompt = ""
        audio_index = 0
        for prompt in prompts:
            if prompt[1] == "speech":
                full_prompt += audio_tokens_str_list[audio_index]
                audio_index += 1
            else:
                full_prompt += prompt[0]["text"]
        ret = self.model.processor([full_prompt])
        predicted_ids = self.model.generate(
            input_ids=ret.input_ids.cuda(),
            attention_mask=ret.attention_mask.cuda(),
            labels=None,
            audios=ret.audios.cuda() if ret.audios is not None else None,
            encoder_length=ret.encoder_length.cuda() if ret.encoder_length is not None else None,
            bridge_length=ret.bridge_length.cuda() if ret.bridge_length is not None else None,
            max_new_tokens=5,
            num_beams=1,
            do_sample=False, 
            top_k=5, 
            top_p=0.85, 
            temperature=0.5,
            num_return_sequences=1,
            repetition_penalty=1.3,
            stop_strings=["\n"],
            tokenizer=self.tokenizer,
        )
        generated = self.tokenizer.batch_decode(predicted_ids[:,ret.input_ids.shape[1]:], skip_special_tokens=True)
        return generated[0]
    
    @torch.no_grad()
    def gen_speech(self, prompts, output_audio_path):
        audio_tokens_str_list = [
            self.tokenize_audio([prompt[0]["speech"]])[0] if prompt[2] == "input" else self.tokenize_audio_gen([prompt[0]["speech"]])[0]
            for prompt in prompts if prompt[1] == "speech"
        ]
        full_prompt = ""
        audio_index = 0
        for prompt in prompts:
            if prompt[1] == "speech":
                full_prompt += audio_tokens_str_list[audio_index]
                audio_index += 1
            else:
                full_prompt += prompt[0]["text"]
        full_prompt += self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audiogen_start_token_id)
        ret = self.model.processor([full_prompt])
        gret = GenerationAudioTokens.generate(
            self.model, 
            input_ids=ret.input_ids.cuda(),
            labels=None,
            audios=ret.audios.cuda() if ret.audios is not None else None,
            encoder_length=ret.encoder_length.cuda() if ret.encoder_length is not None else None,
            bridge_length=ret.bridge_length.cuda() if ret.bridge_length is not None else None,
            past_key_values=None,
            max_new_tokens=25,
            num_beams=1,
            do_sample=False, 
            top_k=5, 
            top_p=0.85, 
            temperature=0.5,
            num_return_sequences=1,
            repetition_penalty=1.3,
            return_dict_in_generate=True,
        )
        decode_save_concat([gret.audios_sequences], self.vocoder, self.model, str(output_audio_path), sampling_rate, wave_concat_overlap)
        