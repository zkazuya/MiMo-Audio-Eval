# Copyright 2025 Xiaomi Corporation.
import tqdm
import json
import os
import string
from pathlib import Path
from zhon.hanzi import punctuation
from jiwer import compute_measures
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel


class InstructTTSEvaluator:

    def __init__(self, model, dataset, model_type, device):
        self.model = model
        self.dataset = dataset
        self.punctuation_all = punctuation + string.punctuation
        self.device = device
        self.model_type = model_type
        if self.dataset.lang == "en":
            self.asr_processor, self.asr_model = self.load_en_asr_model()
        elif self.dataset.lang == "zh":
            self.asr_model = self.load_zh_asr_model()
        else:
            raise NotImplementedError("Language not supported")
    
    def load_en_asr_model(self):
        model_id = "openai/whisper-large-v3"
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)
        return processor, model

    def load_zh_asr_model(self):
        device_str = str(self.device)
        model = AutoModel(model="paraformer-zh", device=device_str)
        return model
    
    def wer(self, hyp, ref):
        for x in self.punctuation_all:
            if x == "'":
                continue
            ref = ref.replace(x, '')
            hyp = hyp.replace(x, '')
        ref = ref.replace('  ', ' ')
        hyp = hyp.replace('  ', ' ')

        if self.dataset.lang == "zh":
            ref = " ".join([x for x in ref])
            hyp = " ".join([x for x in hyp])
        elif self.dataset.lang == "en":
            ref = ref.lower()
            hyp = hyp.lower()
        else:
            raise NotImplementedError("Language not supported")

        measures = compute_measures(ref, hyp)
        wer = measures["wer"]
        return wer, measures

    def asr(self, audio_path):
        if self.dataset.lang == "en":
            wav, sr = sf.read(audio_path)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            input_features = self.asr_processor(wav, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(self.device) 
            forced_decoder_ids = self.asr_processor.get_decoder_prompt_ids(language="english", task="transcribe")
            predicted_ids = self.asr_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = self.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        elif self.dataset.lang == "zh":
            res = self.asr_model.generate(input=audio_path, batch_size_s=300)
            transcription = res[0]["text"]
            transcription = zhconv.convert(transcription, 'zh-cn')
        else:
            raise NotImplementedError("Language not supported")
        return transcription.strip()

    def evaluate(self, output_dir, rank=0, world_size=1):
        assert self.model_type == "instruct", "Base model type not supported"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        wav_dir = output_dir / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)
        results_per_rank = []
        
        dataset_shard = list(self.dataset)[rank::world_size]
        
        if rank == 0:
            print("\n========== Instruct TTS Inference ==========\n")
        
        total_samples = len(dataset_shard)
        progress_bar = tqdm.tqdm(dataset_shard, desc=f"Running Instruct TTS Inference (Rank {rank})", disable=(rank != 0))
        
        for i, data in enumerate(progress_bar):
            ref = data["text"]
            output_audio_path_aps = wav_dir / f"{data['id']}.aps.wav"
            output_audio_path_dsd = wav_dir / f"{data['id']}.dsd.wav"
            output_audio_path_rp = wav_dir / f"{data['id']}.rp.wav"
            
            instructions = {
                "aps": data["APS"],
                "dsd": data["DSD"],
                "rp": data["RP"]
            }
            
            if not output_audio_path_aps.exists():
                self.model.tts_sft(ref, output_audio_path_aps, data["APS"])
            else:
                print(f"{output_audio_path_aps} exists, skipped.")
            
            if not output_audio_path_dsd.exists():
                self.model.tts_sft(ref, output_audio_path_dsd, data["DSD"])
            else:
                print(f"{output_audio_path_dsd} exists, skipped.")
            
            if not output_audio_path_rp.exists():
                self.model.tts_sft(ref, output_audio_path_rp, data["RP"])
            else:
                print(f"{output_audio_path_rp} exists, skipped.")

            if rank == 0:
                print(f"\n----- Sample {i+1}/{total_samples} ID: {data['id']} (Rank {rank}) -----")
                print(f"Reference: {ref}")
                print(f"Audio: {output_audio_path_aps}, {output_audio_path_dsd}, {output_audio_path_rp}")
                print("-" * 70)
                
            results_per_rank.append({
                "id": data["id"],
                "ref": ref,
                "audio_aps": str(output_audio_path_aps.resolve()), 
                "audio_dsd": str(output_audio_path_dsd.resolve()), 
                "audio_rp": str(output_audio_path_rp.resolve()),
                "instruction_aps": instructions["aps"],
                "instruction_dsd": instructions["dsd"],
                "instruction_rp": instructions["rp"]
            })
        
        progress_bar = tqdm.tqdm(results_per_rank, desc=f"Running ASR (Rank {rank})", disable=(rank != 0))
        for data in progress_bar:
            hyp_aps = self.asr(data["audio_aps"])
            hyp_dsd = self.asr(data["audio_dsd"])
            hyp_rp = self.asr(data["audio_rp"])
            wer_aps, measures_aps = self.wer(hyp_aps, data["ref"])
            wer_dsd, measures_dsd = self.wer(hyp_dsd, data["ref"])
            wer_rp, measures_rp = self.wer(hyp_rp, data["ref"])
            data["hyp_aps"] = hyp_aps
            data["hyp_dsd"] = hyp_dsd
            data["hyp_rp"] = hyp_rp
            data["substitutions_aps"] = measures_aps["substitutions"]
            data["insertions_aps"] = measures_aps["insertions"]
            data["deletions_aps"] = measures_aps["deletions"]
            data["substitutions_dsd"] = measures_dsd["substitutions"]
            data["insertions_dsd"] = measures_dsd["insertions"]
            data["deletions_dsd"] = measures_dsd["deletions"]
            data["substitutions_rp"] = measures_rp["substitutions"]
            data["insertions_rp"] = measures_rp["insertions"]
            data["deletions_rp"] = measures_rp["deletions"]
            data["wer_aps"] = wer_aps
            data["wer_dsd"] = wer_dsd
            data["wer_rp"] = wer_rp
        
        with open(output_dir / f"instruct_tts_results_rank_{rank}.json", "w", encoding="utf-8") as f:
            json.dump(results_per_rank, f, indent=4, ensure_ascii=False)
    
        print(f"Rank {rank} Instruct TTS inference complete. Results saved to {output_dir / f'instruct_tts_results_rank_{rank}.json'}")

    def calculate_metrics(self, output_dir, rank=0, world_size=1):
        assert self.model_type == "instruct", "Base model type not supported"
        if rank != 0:
            return None

        output_dir = Path(output_dir)
        all_results = []
        for r in range(world_size):
            rank_file = output_dir / f"instruct_tts_results_rank_{r}.json"
            if rank_file.exists():
                with open(rank_file, "r", encoding="utf-8") as f:
                    all_results.extend(json.load(f))
            else:
                print(f"Warning: Results file {rank_file} not found.")

        avg_wer_aps = sum([result["wer_aps"] for result in all_results]) / len(all_results)
        avg_wer_dsd = sum([result["wer_dsd"] for result in all_results]) / len(all_results)
        avg_wer_rp = sum([result["wer_rp"] for result in all_results]) / len(all_results)
        
        total_subs_aps = sum([result["substitutions_aps"] for result in all_results])
        total_ins_aps = sum([result["insertions_aps"] for result in all_results])
        total_del_aps = sum([result["deletions_aps"] for result in all_results])
        total_subs_dsd = sum([result["substitutions_dsd"] for result in all_results])
        total_ins_dsd = sum([result["insertions_dsd"] for result in all_results])
        total_del_dsd = sum([result["deletions_dsd"] for result in all_results])
        total_subs_rp = sum([result["substitutions_rp"] for result in all_results])
        total_ins_rp = sum([result["insertions_rp"] for result in all_results])
        total_del_rp = sum([result["deletions_rp"] for result in all_results])
        
        summary = {
            "avg_wer_aps": avg_wer_aps,
            "avg_wer_dsd": avg_wer_dsd,
            "avg_wer_rp": avg_wer_rp,
            "total_samples": len(all_results),
            "total_substitutions_aps": total_subs_aps,
            "total_insertions_aps": total_ins_aps,
            "total_deletions_aps": total_del_aps,
            "total_substitutions_dsd": total_subs_dsd,
            "total_insertions_dsd": total_ins_dsd,
            "total_deletions_dsd": total_del_dsd,
            "total_substitutions_rp": total_subs_rp,
            "total_insertions_rp": total_ins_rp,
            "total_deletions_rp": total_del_rp,
            "language": getattr(self.dataset, "lang", "unknown"),
        }
        
        full_results = {
            "summary": summary,
            "results": all_results 
        }
        
        with open(output_dir / "instruct_tts_results.json", "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=4, ensure_ascii=False)
            
        return summary
