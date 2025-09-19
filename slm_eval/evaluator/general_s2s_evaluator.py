# Copyright 2025 Xiaomi Corporation.
import re
import tqdm
import json
import random
import torch
import torchaudio
import string
import os
from zhon.hanzi import punctuation
from pathlib import Path
from jiwer import compute_measures
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel
    

class GeneralS2SEvaluator:

    def __init__(self, model, dataset, n_few_shots=0, seed=42):
        self.model = model
        self.dataset = dataset
        self.n_few_shots = n_few_shots
        self.dataset_name = dataset.dataset_name
        random.seed(seed) 
        self.lang = dataset.dataset_lang
        if self.lang == "en":
            self.en_asr_processor, self.en_asr_model = self.load_en_asr_model()
        elif self.lang == "zh":
            self.zh_asr_model = self.load_zh_asr_model()
        else:
            raise NotImplementedError("Language not supported")
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

    def asr(self, audio_path):
        if self.lang == "en":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            wav, sr = sf.read(audio_path)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            input_features = self.en_asr_processor(wav, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device) 
            forced_decoder_ids = self.en_asr_processor.get_decoder_prompt_ids(language="english", task="transcribe")
            predicted_ids = self.en_asr_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = self.en_asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        elif self.lang == "zh":
            res = self.zh_asr_model.generate(input=audio_path, batch_size_s=300)
            transcription = res[0]["text"]
            transcription = zhconv.convert(transcription, 'zh-cn')
            transcription = ''.join(transcription.split())
            if transcription and not transcription[-1] in ['。', '！', '？', '，', '；']:
                transcription += '。'
        else:
            raise NotImplementedError("Language not supported")
        return transcription.strip()

    def evaluate(self, output_dir, rank=0, world_size=1):
        output_dir = Path(output_dir+f"{self.dataset_name}"+f"_{self.n_few_shots}_shots")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n========== {self.dataset_name} Evaluation (Rank {rank}/{world_size-1}) ==========\n")
        
        total_samples = len(self.dataset)
        samples_per_rank = (total_samples + world_size - 1) // world_size
        start_idx = rank * samples_per_rank
        end_idx = min(start_idx + samples_per_rank, total_samples)
        
        all_prompt_info = []
        
        rank_dataset = [self.dataset[i] for i in range(start_idx, end_idx)]
        progress_bar = tqdm.tqdm(rank_dataset, desc=f"Evaluating {self.dataset_name} (Rank {rank})")
        
        for i, data in enumerate(progress_bar):
            source_path = data["source"]
            target_path = data["target"]
            global_i = start_idx + i
            all_indices = list(range(total_samples))
            
            if self.n_few_shots > 0:
                candidate_indices = all_indices[:global_i] + all_indices[global_i+1:]
                
                k = min(self.n_few_shots, len(candidate_indices))
                sampled_idx = random.sample(candidate_indices, k)
                
                prompt_examples = []
                for j in sampled_idx:
                    sample = self.dataset[j]
                    prompt_source = sample["source"]
                    prompt_target = sample["target"]
                    
                    label_transcription = self.asr(prompt_target)
                    
                    prompt_examples.append({
                        "audio": prompt_source,
                        "label": prompt_target,
                        "label_transcription": label_transcription
                    })
            else:
                prompt_examples = []
            
            source_name = Path(source_path).stem
            target_name = Path(target_path).stem
            
            output_path = output_dir / f"{source_name}_{target_name}.wav"
            
            if prompt_examples:
                last_prompt = prompt_examples[-1]
                ref_src_output_path = output_dir / f"{source_name}_{target_name}_ref_src.wav"
                if os.path.lexists(ref_src_output_path):
                    os.remove(ref_src_output_path)
                os.symlink(os.path.abspath(last_prompt["audio"]), ref_src_output_path)
                ref_tgt_output_path = output_dir / f"{source_name}_{target_name}_ref_tgt.wav"
                if os.path.lexists(ref_tgt_output_path):
                    os.remove(ref_tgt_output_path)
                os.symlink(os.path.abspath(last_prompt["label"]), ref_tgt_output_path)
            
            src_output_path = output_dir / f"{source_name}_{target_name}_src.wav"
            if os.path.lexists(src_output_path):
                os.remove(src_output_path)
            os.symlink(os.path.abspath(source_path), src_output_path)
            
            prompt_info = {
                "target_file": source_path,
                "prompt_files": [self.dataset[idx]["source"] for idx in sampled_idx] if self.n_few_shots > 0 else []
            }
            all_prompt_info.append(prompt_info)
            
            self.model.in_context_learning_s2s(prompt_examples=prompt_examples, audio=source_path, output_path=output_path)
            
        prompt_info_path = output_dir / f"prompt_info_rank{rank}.json"
        with open(prompt_info_path, "w", encoding="utf-8") as f:
            json.dump(all_prompt_info, f, indent=2, ensure_ascii=False)

        return

    def calculate_metrics(self, output_dir, rank=0, world_size=1):
        output_dir = Path(output_dir + f"{self.dataset_name}" + f"_{self.n_few_shots}_shots")
        
        print(f"\n========== {self.dataset_name} Emotion Similarity Calculation (Rank {rank}/{world_size-1}) ==========\n")
        
        generated_files = list(output_dir.glob("*.wav"))
        main_files = [f for f in generated_files if not f.stem.endswith("_ref")]
        
        files_per_rank = (len(main_files) + world_size - 1) // world_size
        start_idx = rank * files_per_rank
        end_idx = min(start_idx + files_per_rank, len(main_files))
        rank_files = main_files[start_idx:end_idx]
        
        total_emotion_sim = 0
        total_samples = 0
        valid_pairs = 0
        
        progress_bar = tqdm.tqdm(rank_files, desc=f"Calculating emotion similarity (Rank {rank})")
        
        for generated_file in progress_bar:
            ref_file = output_dir / f"{generated_file.stem}_ref.wav"
            
            if not ref_file.exists():
                print(f"Warning: Reference file not found for {generated_file.name}")
                continue
            
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                xs = [str(ref_file)]
                x_hats = [str(generated_file)]
                
                sim_score, n_samples = self.emotion_sim_evaluator.forward(xs, x_hats, device)
                
                if n_samples > 0:
                    avg_sim = sim_score / n_samples
                    total_emotion_sim += sim_score
                    total_samples += n_samples
                    valid_pairs += 1
                    
                    progress_bar.set_postfix({
                        "avg_emotion_sim": total_emotion_sim / total_samples if total_samples > 0 else 0,
                        "valid_pairs": valid_pairs
                    })
                else:
                    print(f"Warning: Failed to compute emotion similarity for {generated_file.name}")
                    
            except Exception as e:
                print(f"Error processing {generated_file.name}: {e}")
                continue
        
        if rank == 0 and total_samples > 0:
            avg_emotion_sim = total_emotion_sim / total_samples
            
            print(f"\nFinal Emotion Similarity: {avg_emotion_sim:.4f}")
            print(f"Valid pairs processed: {valid_pairs}")
            print(f"Total samples: {total_samples}")
            
            results = {
                "emotion_similarity": float(avg_emotion_sim),
                "valid_pairs": int(valid_pairs),
                "total_samples": int(total_samples),
                "n_few_shots": self.n_few_shots,
                "dataset_name": self.dataset_name,
                "dataset_lang": self.lang
            }
            
            results_file = output_dir / "emotion_similarity_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {results_file}")
        
        return {
            "emotion_similarity": total_emotion_sim / total_samples if total_samples > 0 else 0,
            "valid_pairs": valid_pairs,
            "total_samples": total_samples
        }