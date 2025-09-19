# Copyright 2025 Xiaomi Corporation.
import re
import os
import tqdm
import json
import random
import torchaudio
import tempfile
import numpy as np

from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import scipy
from funasr import AutoModel
import zhconv

PUNCTUATION_REGEX = re.compile(r"[^\w\s]")


class DynamicSuperbS2SEvaluator:

    def __init__(self, model, fused_dataset, n_few_shots=0, seed=42):
        self.model = model
        dataset, dataset_name, sample_rate = fused_dataset
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.sample_rate = sample_rate   
        
        self.n_few_shots = n_few_shots
        random.seed(seed)
        self.en_asr_processor, self.en_asr_model = self.load_en_asr_model()
        self.zh_asr_model = self.load_zh_asr_model()
        
        self.sample_categories = {}
            
        if dataset_name == "DynamicSuperb/SuperbSE_VoiceBankDEMAND-Test":
            for idx, sample in enumerate(self.dataset):
                speaker_id = sample["file"].split("_")[0]
                
                if speaker_id not in self.sample_categories:
                    self.sample_categories[speaker_id] = []
                self.sample_categories[speaker_id].append(idx)
            
            for speaker_id, indices in self.sample_categories.items():
                if len(indices) <= n_few_shots:
                    print(f"Warning: Speaker {speaker_id} has {len(indices)} samples, which is not enough for {n_few_shots} prompt examples")
        else:
            raise NotImplementedError

    def load_en_asr_model(self):
        model_id = "openai/whisper-large-v3"
        processor = WhisperProcessor.from_pretrained(model_id)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
        return processor, model

    def load_zh_asr_model(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device_str = str(device)
        model = AutoModel(model="paraformer-zh", device=device_str)
        return model

    def asr(self, audio, lang="en"):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if lang == "zh":
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio)
                audio = audio.astype(np.float32)
                resampled_audio = scipy.signal.resample(audio, int(len(audio) * 16000 / self.sample_rate))
                torchaudio.save(temp_file.name, torch.from_numpy(resampled_audio).unsqueeze(0), 16000)
                res = self.zh_asr_model.generate(input=temp_file.name, batch_size_s=300)
                transcription = res[0]["text"]
                transcription = zhconv.convert(transcription, 'zh-cn')
                transcription = ''.join(transcription.split())
                if transcription and not transcription[-1] in ['。', '！', '？', '，', '；']:
                    transcription += '。'
        else:
            resampled_audio = scipy.signal.resample(audio, int(len(audio) * 16000 / self.sample_rate))
            
            input_features = self.en_asr_processor(resampled_audio, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device)
            
            forced_decoder_ids = self.en_asr_processor.get_decoder_prompt_ids(language="english", task="transcribe")
            predicted_ids = self.en_asr_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = self.en_asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription.strip()

    def evaluate(self, output_dir, rank=0, world_size=1):
        output_dir = Path(output_dir+f"_{self.n_few_shots}_shots")
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        
        all_prompt_info = []
        
        if rank == 0:
            print(f"\n========== {self.dataset_name} Evaluation ==========\n")
            print(f"Total samples: {len(self.dataset)}, Processes: {world_size}")
            print(f"Each process handles approximately {len(self.dataset) // world_size} samples")
        
        
        total_samples = len(self.dataset)
        samples_per_rank = (total_samples + world_size - 1) // world_size
        start_idx = rank * samples_per_rank
        end_idx = min(start_idx + samples_per_rank, total_samples)
        
        rank_dataset = [self.dataset[i] for i in range(start_idx, end_idx)]
        all_indices = list(range(total_samples))
        
        print(f"Rank {rank} processing samples {start_idx} to {end_idx-1} (共 {len(rank_dataset)} 个样本)")
        
        progress_bar = tqdm.tqdm(rank_dataset, desc=f"Evaluating {self.dataset_name} (Rank {rank})")
        
        for local_i, data in enumerate(progress_bar):
            global_i = start_idx + local_i
            
            audio = torch.tensor(data["audio"]["array"]).float()
            
            if self.n_few_shots > 0:
                if self.dataset_name == "DynamicSuperb/SuperbSE_VoiceBankDEMAND-Test":
                    current_speaker_id = data["file"].split("_")[0]
            
                    candidate_indices = [idx for idx in self.sample_categories[current_speaker_id] if idx != global_i]
                    
                    if len(candidate_indices) < self.n_few_shots:
                        print(f"Warning: The speaker ({current_speaker_id}) of the current sample ({data['file']}) has {len(candidate_indices)} available samples, which is not enough for {self.n_few_shots} prompt examples")
                else:
                    raise NotImplementedError
                
                k = min(self.n_few_shots, len(candidate_indices))
                sampled_idx = random.sample(candidate_indices, k)

                prompt_examples = []
                for j in sampled_idx:
                    sample = self.dataset[j]
                    label_transcription = sample.get("label_transcription")
                    label_audio = sample["label"]["array"]
                    
                    if label_transcription is None:
                        lang = "en"
                        label_transcription = self.asr(label_audio, lang=lang)
                        
                    
                    prompt_examples.append({
                        "audio": torch.tensor(sample["audio"]["array"]).float(), 
                        "label": torch.tensor(label_audio).float(),
                        "label_transcription": label_transcription
                    })
            else:
                prompt_examples = []
            
            output_path = output_dir / f"{data['file']}.wav"
            
            if self.dataset_name == "DynamicSuperb/SuperbSE_VoiceBankDEMAND-Test":
                src_path = output_dir / f"{data['file']}_src.wav"
                if not isinstance(audio, np.ndarray):
                    audio_np = np.array(audio)
                else:
                    audio_np = audio
                audio_np = audio_np.astype(np.float32)
                torchaudio.save(src_path, torch.from_numpy(audio_np).unsqueeze(0), self.sample_rate)
            
            prompt_info = {
                "target_file": data["file"],
                "prompt_files": [self.dataset[idx]["file"] for idx in sampled_idx] if self.n_few_shots > 0 else []
            }
            all_prompt_info.append(prompt_info)
            
            self.model.in_context_learning_s2s(
                prompt_examples=prompt_examples,
                audio=audio,
                output_path=output_path
            )
            
            results.append({
                "file": str(output_path),
                "original_file": data['file']
            })
        
        prompt_info_path = output_dir / f"prompt_info_rank{rank}.json"
        with open(prompt_info_path, "w", encoding="utf-8") as f:
            json.dump(all_prompt_info, f, indent=2, ensure_ascii=False)
        
        return {
            "processed_files": len(results),
            "results": results,
            "rank": rank,
            "total_ranks": world_size
        }
    