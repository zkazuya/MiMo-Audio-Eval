# Copyright 2025 Xiaomi Corporation.
from pathlib import Path
import torch
import json
import os
import torch.distributed as dist
import tqdm
import zhconv
import whisper
from openai import OpenAI


class BigBenchAudioEvaluator:

    def __init__(self, model, dataset, task, model_type=None, device=None):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model_type = model_type
        self.eval_mode = task.split("_")[-1]  # "s2t" or "s2s"
        self.judge_model = self._init_judge_model()
        self.prompt_speech_path = Path(os.getcwd()) / "assets" / "prompt.wav"
        
        if self.eval_mode == "s2s":
            self.asr_model = self.load_en_asr_model()

    def _init_judge_model(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        return OpenAI(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
        )

    def load_en_asr_model(self):
        model = whisper.load_model("large-v3").to(self.device)
        return model

    def asr(self, audio_path):
        transcription = self.asr_model.transcribe(audio_path)["text"]
        return transcription.strip()

    def evaluate_one_sample(self, answer, prediction):
        user_content = f"Model Response: {prediction}\nReference Answer: {answer}"
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                completion = self.judge_model.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": 'You are a helpful assistant that evaluates whether the model\'s response matches the provided reference answer. Respond ONLY with a single JSON object matching: {"steps":[{"description":"string","conclusion":"string"}],"result":number}. Do not include any extra text. result must be a float in [0.0, 1.0].'
                        },
                        {
                            "role": "user", 
                            "content": user_content
                        }
                    ]
                )

                response = completion.choices[0].message.content.strip()
                result_data = json.loads(response)
                score = result_data["result"]
                return score
            except:
                print(f"Retrying ... {attempt + 1}/{max_retries}")
                
        return 0.0

    def evaluate(self, output_dir, rank=0, world_size=1):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.eval_mode == "s2s":
            (output_dir / "wav").mkdir(parents=True, exist_ok=True)
        
        predictions_file_path_rank = output_dir / f"predictions_rank_{rank}.json"
        
        predictions_this_rank = []
        processed_item_ids_this_rank = set()

        if predictions_file_path_rank.exists():
            try:
                with open(predictions_file_path_rank, "r", encoding="utf-8") as f:
                    predictions_this_rank = json.load(f)
                for pred_item in predictions_this_rank:
                    processed_item_ids_this_rank.add(pred_item["sample_id"])
                if processed_item_ids_this_rank:
                    print(f"Rank {rank} resumed from checkpoint: {predictions_file_path_rank}. {len(processed_item_ids_this_rank)} items already processed for this rank.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load predictions file {predictions_file_path_rank} for rank {rank}: {e}. Starting fresh inference for this rank.")
                predictions_this_rank = []
                processed_item_ids_this_rank = set()
        
        dataset_shard = list(self.dataset)[rank::world_size]
        items_to_process_this_run = []
        
        for idx, item in enumerate(dataset_shard):
            item_id = rank + idx * world_size
            if item_id not in processed_item_ids_this_rank:
                item['sample_id'] = item_id
                items_to_process_this_run.append(item)
        
        if rank == 0:
            print(f"\n========== Running BigBench Audio Evaluation ==========\n")

        progress_bar = tqdm.tqdm(items_to_process_this_run, desc=f"Running BigBench Audio Evaluation (Rank {rank})", disable=(rank != 0))

        for item in progress_bar:
            print(f"\n----- Sample {item['sample_id']} (Rank {rank}) -----")
            audio = torch.tensor(item["audio"]["array"]).unsqueeze(0).float()
            if self.device:
                audio = audio.to(self.device)
            category = item["category"]
            answer = item["official_answer"]
            
            if self.eval_mode == "s2t":
                response = self.model.speech2text_dialogue_sft(audio)
            elif self.eval_mode == "s2s":
                output_audio_path = output_dir / "wav" / f"{item['sample_id']}.wav"
                response = self.model.spoken_dialogue_sft(audio, output_audio_path=str(output_audio_path), prompt_speech=prompt_speech_path)
                
            if response is not None:
                print(f"Category: {category}")
                print(f"Answer: {answer}")
                print(f"Response: {response}")
                print("-" * 70)
                result_dict = {
                    "sample_id": item["sample_id"],
                    "category": category,
                    "answer": answer,
                    "response": response,
                    "asr_response": None
                }
                predictions_this_rank.append(result_dict)

                temp_file_path_rank = output_dir / f"predictions_rank_{rank}.json.tmp"
                with open(temp_file_path_rank, "w", encoding="utf-8") as f_tmp:
                    json.dump(predictions_this_rank, f_tmp, indent=4, ensure_ascii=False)
                os.replace(temp_file_path_rank, predictions_file_path_rank)
        
        print(f"Rank {rank} BigBench Audio evaluation complete. Processed {len(items_to_process_this_run)} new items. Results saved to {predictions_file_path_rank}")

        with open(predictions_file_path_rank, "r", encoding="utf-8") as f:
            predictions_this_rank = json.load(f)

        if self.eval_mode == "s2s":
            for item in predictions_this_rank:
                output_audio_path = output_dir / "wav" / f"{item['sample_id']}.wav"
                if item['asr_response'] is None:
                    item['asr_response'] = self.asr(str(output_audio_path))
                    print("response: ", item['response'])
                    print("asr_response: ", item['asr_response'])
                    print("-" * 70)

            with open(predictions_file_path_rank, "w", encoding="utf-8") as f:
                json.dump(predictions_this_rank, f, indent=4, ensure_ascii=False)

    def calculate_metrics(self, output_dir, rank=0, world_size=1):
        if rank != 0:   
            return None
        
        output_dir = Path(output_dir)
        all_predictions = []

        predictions_file_path = output_dir / "results.json"
        
        for r in range(world_size):
            predictions_file_path_rank = output_dir / f"predictions_rank_{r}.json"
            if predictions_file_path_rank.exists():
                with open(predictions_file_path_rank, "r", encoding="utf-8") as f:
                    all_predictions.extend(json.load(f))
            else:
                print(f"Warning: Predictions file not found for rank {r} at {predictions_file_path_rank}")
        
        print(f"{len(all_predictions)} predictions for evaluation.")

        correct_cnt = 0
        for item in tqdm.tqdm(all_predictions):
            answer = item['answer']
            pred = item['asr_response'] if self.eval_mode == "s2s" and "asr_response" in item else item['response']
            item['eval_score'] = self.evaluate_one_sample(answer, pred)
            item['correct'] = item['eval_score'] >= 0.6
            if item['correct']:
                correct_cnt += 1
            
            print("Answer: ", answer)
            print("Pred: ", pred)
            print("Eval Score: ", item['eval_score'])
            print("Correct: ", item['correct'])
            print("-" * 70)
        
        accuracy = correct_cnt / len(all_predictions) * 100
        
        results = {
            'acc': accuracy,
            'total': len(all_predictions),
            'correct': correct_cnt,
            'results': all_predictions
        }
        
        with open(predictions_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print("Accuracy: ", accuracy)
        print("Correct: ", correct_cnt)
        print("Total: ", len(all_predictions))
        print("Results saved to: ", predictions_file_path)
        print("=" * 70)

        return results
