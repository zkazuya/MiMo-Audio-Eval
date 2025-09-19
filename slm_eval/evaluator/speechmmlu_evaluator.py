# Copyright 2025 Xiaomi Corporation.
import os
import re
import json
import scipy
import zhconv
import soundfile as sf
from pathlib import Path
import tqdm
from collections import defaultdict
from funasr import AutoModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class SpeechMMLUEvaluator:
    
    def __init__(self, model, dataset, task, n_few_shots=0, device=None):
        self.model = model
        self.dataset = dataset
        self.task = task[-3:]
        assert self.task in ["s2t", "s2s", "t2t", "t2s"], f"Invalid task: {task}. Supported tasks are: speechmmlu_s2t, speechmmlu_s2s, speechmmlu_t2t, speechmmlu_t2s"
        self.n_few_shots = n_few_shots
        self.device = device
        if self.task[-1] == "s":
            if self.dataset.lang == "en":
                self.asr_processor, self.asr_model = self.load_en_asr_model()
            elif self.dataset.lang == "zh":
                self.asr_model = self.load_zh_asr_model()

    def load_en_asr_model(self):
        model_id = "openai/whisper-large-v3"
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)
        return processor, model

    def load_zh_asr_model(self):
        device_str = str(self.device)
        model = AutoModel(model="paraformer-zh", device=device_str)
        return model
    
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
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        predictions_file_path_rank = output_dir / f"predictions_rank_{rank}.json"
        if self.task[-1] == "s": 
            (output_dir / "audio").mkdir(exist_ok=True, parents=True) 
        
        predictions_this_rank = []
        processed_item_ids_this_rank = set()

        if predictions_file_path_rank.exists():
            try:
                with open(predictions_file_path_rank, "r", encoding="utf-8") as f:
                    predictions_this_rank = json.load(f)
                for pred_item in predictions_this_rank:
                    processed_item_ids_this_rank.add(pred_item["id"])
                if processed_item_ids_this_rank:
                    print(f"Rank {rank} resumed from checkpoint: {predictions_file_path_rank}. {len(processed_item_ids_this_rank)} items already processed for this rank.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load predictions file {predictions_file_path_rank} for rank {rank}: {e}. Starting fresh inference for this rank.")
                predictions_this_rank = []
                processed_item_ids_this_rank = set()
        
        dataset_shard = list(self.dataset.data)[rank::world_size]
        items_to_process_this_run = [item for item in dataset_shard if item["id"] not in processed_item_ids_this_rank]
        
        if rank == 0:
            print(f"\n========== Running Speech MMLU Inference ==========\n")

        progress_bar = tqdm.tqdm(items_to_process_this_run, desc=f"Running Speech MMLU Inference (Rank {rank})", disable=(rank != 0))

        for item in progress_bar:
            prediction = self._infer_item(item, output_dir / "audio") 
            predictions_this_rank.append(prediction)

            if rank == 0:
                print("-" * 100)
                print("id", item["id"])
                print("question_and_choices", item["question_and_choices"])
                print("outputs", prediction["outputs"])
                print("model_prediction_idx", prediction["model_prediction_idx"])
                print("correct_answer_idx", prediction["correct_answer_idx"])
                if prediction["model_prediction_idx"] == prediction["correct_answer_idx"]:
                    print("✅ Correct")
                else:
                    print("❌ Incorrect")
                print("-" * 100)
                

            temp_file_path_rank = output_dir / f"predictions_rank_{rank}.json.tmp"
            with open(temp_file_path_rank, "w", encoding="utf-8") as f_tmp:
                json.dump(predictions_this_rank, f_tmp, indent=4, ensure_ascii=False)
            os.replace(temp_file_path_rank, predictions_file_path_rank)

        if self.task[-1] == "s":
            print(f"Rank {rank}: Running ASR on generated audio files...")
            progress_bar = tqdm.tqdm(predictions_this_rank, desc=f"Running ASR (Rank {rank})", disable=(rank != 0))
            for prediction in progress_bar:
                audio_file_path = output_dir / "audio" / f"{prediction['id']}.wav"
                if audio_file_path.exists():
                    asr_text = self.asr(str(audio_file_path))
                    prediction["output_audio_path"] = str(audio_file_path)
                    prediction["asr_text"] = asr_text
                    prediction["model_prediction_idx"] = self.get_answer_from_output_text(asr_text)
                else:
                    print(f"Warning: Audio file not found at {audio_file_path}")
                    prediction["asr_text"] = ""
                    prediction["model_prediction_idx"] = -1

            with open(predictions_file_path_rank, "w", encoding="utf-8") as f:
                json.dump(predictions_this_rank, f, indent=4, ensure_ascii=False)
        
        print(f"Rank {rank} Speech MMLU inference complete. Processed {len(items_to_process_this_run)} new items. Results saved to {predictions_file_path_rank}")

    def calculate_metrics(self, output_dir, rank=0, world_size=1):
        if rank != 0:
            return None
        
        output_dir = Path(output_dir)
        all_predictions = []
        for r in range(world_size):
            predictions_file_path_rank = output_dir / f"predictions_rank_{r}.json"
            if predictions_file_path_rank.exists():
                with open(predictions_file_path_rank, "r", encoding="utf-8") as f:
                    all_predictions.extend(json.load(f))
            else:
                print(f"Warning: Predictions file not found for rank {r} at {predictions_file_path_rank}")
        
        predictions_file_path = output_dir / "predictions.json"
        with open(predictions_file_path, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, indent=4, ensure_ascii=False)
        
        subject_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        total_correct = 0
        total_items = 0
        
        for prediction in tqdm.tqdm(all_predictions, desc="Calculating metrics"):
            prediction["correct"] = prediction["model_prediction_idx"] == prediction["correct_answer_idx"]
            
            subject = prediction["subject"]
            subject_metrics[subject]["total"] += 1
            if prediction["correct"]:
                subject_metrics[subject]["correct"] += 1
            
            total_items += 1
            if prediction["correct"]:
                total_correct += 1
        
        subject_accuracies = {}
        for subject, counts in subject_metrics.items():
            accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            subject_accuracies[subject] = accuracy
        
        overall_accuracy = total_correct / total_items if total_items > 0 else 0
        
        summary = {
            "task": self.task,
            "overall_accuracy": overall_accuracy,
            "subject_accuracies": subject_accuracies,
            "total_correct": total_correct,
            "total_items": total_items,
        }
        
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)
        
        print("\n========== Evaluation Summary ==========\n")    
        print(f"Overall accuracy: {overall_accuracy:.4f} ({total_correct}/{total_items})")
        print("\nSubject Accuracies:")
        for subj, acc in sorted(subject_accuracies.items()):
            print(f"{subj}: {acc:.4f} ({subject_metrics[subj]['correct']}/{subject_metrics[subj]['total']})")

        return summary
    
    def _infer_item(self, item, audio_output_dir):
        """Run inference for a single item and return the prediction"""
        few_shot_prompts = self.dataset.get_few_shot_prompts(item["subject"], item["voice_id"])
        question_and_choices = item["question_and_choices"]
        correct_answer_idx = item["answer"]
        input_mode, output_mode = self.task[0], self.task[-1]

        prompts = self.get_prompt(few_shot_prompts, question_and_choices, input_mode, output_mode)
        
        if output_mode == "t":
            outputs = self.model.gen_text(prompts)
            model_prediction = self.get_answer_from_output_text(outputs)
        else:
            output_path = audio_output_dir / f"{item['id']}.wav"
            outputs = self.model.gen_speech(prompts, output_path)
            model_prediction = -1  # Will be processed later during evaluation
        
        prediction = {
            "id": item["id"],
            "subject": item["subject"],
            "question_and_choices": question_and_choices["text"],
            "correct_answer_idx": correct_answer_idx,
            "model_prediction_idx": model_prediction,
            "outputs": outputs,
        }

        return prediction
    
    def get_prompt(self, few_shot_prompts, question_and_choices, input_mode='t', output_mode='t'):
        input_key = "speech" if input_mode == 's' else "text"
        output_key = "speech" if output_mode == 's' else "text"

        prompts = []
        prompts.append((few_shot_prompts["beginning"], input_key, "input"))

        examples = []
        for i in range(self.n_few_shots):
            e = few_shot_prompts["examples"][i]
            examples.append((e["question_and_choices"], input_key, "input"))
            examples.append((e["answer"], output_key, "output"))

        prompts.extend(examples)
        prompts.append((question_and_choices, input_key, "input"))

        return prompts
    
    def get_answer_from_output_text(self, outputs):
        if self.dataset.lang == "en":
            outputs = outputs.lower()
            match = re.search(r"option\s*([abcd])", outputs)
        else:
            outputs = outputs.replace(" ", "").lower()
            match = re.search(r"选项\s*([abcd])", outputs)
        if match:
            option_letter = match.group(1)
            mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
            return mapping[option_letter]
        else:
            return -1