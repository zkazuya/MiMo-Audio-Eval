# Copyright 2025 Xiaomi Corporation.
import os
import tqdm
import json
import random
import torch
import torchaudio
import string
from zhon.hanzi import punctuation
from pathlib import Path
from .speaker_sim import SpeakerSimEvaluator
from jiwer import compute_measures
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel

SIM_MODEL_PATH = Path(os.getcwd()) / "data/wavlm_large_finetune.pth"


class VoiceConversionS2SEvaluator:

    def __init__(self, model, dataset, n_few_shots=0, seed=42):
        self.model = model
        self.dataset = dataset
        self.n_few_shots = n_few_shots
        self.dataset_name = "ESD"
        random.seed(seed) 
        self.speaker_sim_evaluator = SpeakerSimEvaluator(model_path=SIM_MODEL_PATH, sampling_rate=16000)
        self.en_asr_processor, self.en_asr_model = self.load_en_asr_model()
        self.zh_asr_model = self.load_zh_asr_model()
        self.punctuation_all = string.punctuation + punctuation

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
    
    def wer(self, hyp, ref, lang):
        for x in self.punctuation_all:
            if x == "'":
                continue
            ref = ref.replace(x, '')
            hyp = hyp.replace(x, '')
        ref = ref.replace('  ', ' ')
        hyp = hyp.replace('  ', ' ')

        if lang == "zh":
            ref = " ".join([x for x in ref])
            hyp = " ".join([x for x in hyp])
        elif lang == "en":
            ref = ref.lower()
            hyp = hyp.lower()
        else:
            raise NotImplementedError("Language not supported")

        measures = compute_measures(ref, hyp)
        wer = measures["wer"]
        return wer, measures

    def asr(self, audio_path, lang):
        if lang == "en":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            wav, sr = sf.read(audio_path)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            input_features = self.en_asr_processor(wav, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device) 
            forced_decoder_ids = self.en_asr_processor.get_decoder_prompt_ids(language="english", task="transcribe")
            predicted_ids = self.en_asr_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = self.en_asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        elif lang == "zh":
            res = self.zh_asr_model.generate(input=audio_path, batch_size_s=300)
            transcription = res[0]["text"]
            transcription = zhconv.convert(transcription, 'zh-cn')
        else:
            raise NotImplementedError("Language not supported")
        return transcription.strip()

    def infer_language(self, text):
        chinese_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha() or '\u4e00' <= char <= '\u9fff':
                total_chars += 1
                if '\u4e00' <= char <= '\u9fff':
                    chinese_chars += 1
        
        if total_chars == 0:
            return "en"
        
        chinese_ratio = chinese_chars / total_chars
        if chinese_ratio > 0.5:
            return "zh"
        else:
            return "en"

    def get_ref_text(self, ref_id):
        speaker_id = ref_id.split('_')[0]
        text_file_path = os.path.join(os.getcwd(), f"data/ESD/proc/EmotionSpeechDataset/{speaker_id}/{speaker_id}.txt")
        
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2 and parts[0] == ref_id:
                        return parts[1]
        except FileNotFoundError:
            print(f"Warning: Text file not found: {text_file_path}")
        except Exception as e:
            print(f"Warning: Error reading text file {text_file_path}: {e}")
        
        return None

    def evaluate(self, output_dir, rank=0, world_size=1):
        output_dir = Path(output_dir+f"_{self.n_few_shots}_shots")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n========== {self.dataset_name} Evaluation (Rank {rank}/{world_size-1}) ==========\n")
        
        total_samples = len(self.dataset)
        samples_per_rank = (total_samples + world_size - 1) // world_size
        start_idx = rank * samples_per_rank
        end_idx = min(start_idx + samples_per_rank, total_samples)
        
        rank_dataset = [self.dataset[i] for i in range(start_idx, end_idx)]
        progress_bar = tqdm.tqdm(rank_dataset, desc=f"Evaluating {self.dataset_name} (Rank {rank})")
        
        for i, data in enumerate(progress_bar):
            audio = data["source"]

            prompt_examples = [{"audio": example["source"], 
                                "label": example["target"],
                                "label_transcription": example["target_text"]} for example in data["prompt"]]
            
            subdir = output_dir / f"{data['source_id']}_{data['target_id']}"
            subdir.mkdir(parents=True, exist_ok=True)
            
            os.symlink(os.path.abspath(audio), subdir / "source.wav")
            
            if prompt_examples:
                def get_ids_from_path(path):
                    parts = os.path.basename(path).split('_')
                    if len(parts) >= 2:
                        return f"{parts[0]}_{parts[1]}"
                    return "unknown"
                
                first_source_id = get_ids_from_path(prompt_examples[0]["audio"])
                first_target_id = get_ids_from_path(prompt_examples[0]["label"])
                os.symlink(os.path.abspath(prompt_examples[0]["audio"]), 
                          subdir / f"prompt_first_source_{first_source_id}.wav")
                os.symlink(os.path.abspath(prompt_examples[0]["label"]), 
                          subdir / f"prompt_first_target_{first_target_id}.wav")
                
                last_source_id = get_ids_from_path(prompt_examples[-1]["audio"])
                last_target_id = get_ids_from_path(prompt_examples[-1]["label"])
                os.symlink(os.path.abspath(prompt_examples[-1]["audio"]), 
                          subdir / f"prompt_last_source_{last_source_id}.wav")
                os.symlink(os.path.abspath(prompt_examples[-1]["label"]), 
                          subdir / f"prompt_last_target_{last_target_id}.wav")
            
            output_path = subdir / "generated.wav"
            self.model.in_context_learning_s2s(prompt_examples=prompt_examples, audio=audio, output_path=output_path)
        
        print(f"Rank {rank} evaluation complete. Processed {len(rank_dataset)} samples.")

    def calculate_metrics(self, output_dir, rank=0, world_size=1):
        if world_size !=1:
            raise NotImplementedError("Only support world_size=1")
        output_dir = Path(output_dir+f"_{self.n_few_shots}_shots")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        subdirs_per_rank = (len(all_subdirs) + world_size - 1) // world_size
        start_idx = rank * subdirs_per_rank
        end_idx = min(start_idx + subdirs_per_rank, len(all_subdirs))
        rank_subdirs = all_subdirs[start_idx:end_idx]
        
        total_sim_score = 0
        total_samples = 0
        total_wer_zh = 0
        wer_samples_zh = 0
        total_wer_en = 0
        wer_samples_en = 0
        
        progress_bar = tqdm.tqdm(rank_subdirs, desc=f"Calculating metrics (Rank {rank})")
        
        for subdir in progress_bar:
            generated_wav = subdir / "generated.wav"
            if not generated_wav.exists():
                continue
                
            prompt_last_target_files = list(subdir.glob("prompt_last_target_*.wav"))
            if not prompt_last_target_files:
                continue
                
            parts = subdir.name.split('_')
            if len(parts) >= 4:
                target_ref_id = f"{parts[2]}_{parts[3]}"
                
                gen_audio, sr = torchaudio.load(str(generated_wav))
                if sr != 16000:
                    gen_audio = torchaudio.transforms.Resample(sr, 16000)(gen_audio)
                
                ref_audio, sr = torchaudio.load(str(prompt_last_target_files[0]))
                if sr != 16000:
                    ref_audio = torchaudio.transforms.Resample(sr, 16000)(ref_audio)

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                sim_score, n_samples = self.speaker_sim_evaluator([ref_audio], [gen_audio], device=device)
                total_sim_score += sim_score
                total_samples += n_samples
                
                ref_text = self.get_ref_text(target_ref_id)
                if ref_text is not None:
                    lang = self.infer_language(ref_text)
                    gen_transcription = self.asr(str(generated_wav), lang)
                    wer_score, _ = self.wer(gen_transcription, ref_text, lang)
                    
                    if lang == "zh":
                        total_wer_zh += wer_score
                        wer_samples_zh += 1
                    elif lang == "en":
                        total_wer_en += wer_score
                        wer_samples_en += 1
                    
                    progress_bar.set_postfix({
                        "avg_sim": total_sim_score / total_samples if total_samples > 0 else 0,
                        "wer_zh": total_wer_zh / wer_samples_zh if wer_samples_zh > 0 else 0,
                        "wer_en": total_wer_en / wer_samples_en if wer_samples_en > 0 else 0,
                        "zh_samples": wer_samples_zh,
                        "en_samples": wer_samples_en
                    })
                else:
                    progress_bar.set_postfix({"avg_sim": total_sim_score / total_samples if total_samples > 0 else 0})
            
        if rank == 0:
            avg_sim = total_sim_score / total_samples if total_samples > 0 else 0
            avg_wer_zh = total_wer_zh / wer_samples_zh if wer_samples_zh > 0 else 0
            avg_wer_en = total_wer_en / wer_samples_en if wer_samples_en > 0 else 0
            
            print(f"\nFinal Speaker Similarity: {avg_sim:.4f}")
            if wer_samples_zh > 0:
                print(f"Final WER (Chinese): {avg_wer_zh:.4f} ({wer_samples_zh} samples)")
            if wer_samples_en > 0:
                print(f"Final WER (English): {avg_wer_en:.4f} ({wer_samples_en} samples)")
            
            results = {
                "speaker_similarity": float(avg_sim),
                "wer_zh": float(avg_wer_zh) if wer_samples_zh > 0 else None,
                "wer_en": float(avg_wer_en) if wer_samples_en > 0 else None,
                "processed_samples": int(total_samples),
                "wer_samples_zh": int(wer_samples_zh),
                "wer_samples_en": int(wer_samples_en),
                "n_few_shots": self.n_few_shots,
                "dataset_name": self.dataset_name
            }
            
            results_file = output_dir / "evaluation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {results_file}")
            
            txt_file = output_dir / "evaluation_results.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"Voice Conversion Evaluation Results\n")
                f.write(f"=" * 40 + "\n\n")
                f.write(f"Dataset: {self.dataset_name}\n")
                f.write(f"Few-shot examples: {self.n_few_shots}\n\n")
                f.write(f"Speaker Similarity: {avg_sim:.4f} ({total_samples} samples)\n")
                if wer_samples_zh > 0:
                    f.write(f"WER (Chinese): {avg_wer_zh:.4f} ({wer_samples_zh} samples)\n")
                if wer_samples_en > 0:
                    f.write(f"WER (English): {avg_wer_en:.4f} ({wer_samples_en} samples)\n")
            print(f"Results also saved to: {txt_file}")
        
        return {
            "speaker_similarity": total_sim_score / total_samples if total_samples > 0 else 0,
            "wer_zh": total_wer_zh / wer_samples_zh if wer_samples_zh > 0 else 0,
            "wer_en": total_wer_en / wer_samples_en if wer_samples_en > 0 else 0,
            "processed_samples": total_samples,
            "wer_samples_zh": wer_samples_zh,
            "wer_samples_en": wer_samples_en
        }