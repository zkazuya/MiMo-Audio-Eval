# Copyright 2025 Xiaomi Corporation.
import os
import json
import tqdm
from pathlib import Path
import json
from typing import Literal, Any
from pydantic import BaseModel
from abc import ABC, abstractmethod
from openai import OpenAI
import whisper


class OpenAIModel:
    
    def __init__(self, model: str, temp: float, response_format: Any = None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
                 
        self.client = OpenAI(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
        )

        self.model = 'gpt-4o'
        self.temp = float(temp)
        self.response_format = response_format or False

    def generate(self, prompt:Any):
        if type(prompt) == str:
            prompt = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) and 'role' in item and item['role'] in ['user', 'assistant'] for item in prompt):
            pass 
        else:
            raise ValueError("Prompt must be a string or a list of dictionaries with 'role' keys as 'user' or 'assistant'.")
        
        if self.response_format:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=prompt,
                temperature=self.temp,
                response_format=self.response_format,
                timeout=100.0
            )
            return response.choices[0].message.parsed
        else:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = prompt,
                temperature = self.temp,
                timeout=100.0
            )
            return response.choices[0].message.content


class JudgeResponse(BaseModel):
    reasoning: str
    verdict: Literal["YES", "NO"]

JUDGE_PROMPT = '''You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{}
</CRITERIA>

Print your reasoning followed by your verdict, either "YES" or "NO".'''


class MultiChallengeEvaluator:
    
    def __init__(self, model, dataset, task, device=None, model_type='base', n_few_shots=5, exec_mode='infer', thinking=False):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model_type = model_type
        self.thinking = thinking
        root_path = os.getcwd()
        self.prompt_root_path = os.path.join(root_path, "data/multi_challenge/prompt_wav")
        self.evaluation_model = OpenAIModel(
            model="gpt-4o",
            temp=0, 
            response_format=JudgeResponse
        )
        
        self.task_type = task.split("_")[-1]

        if self.task_type == 's2s':
            self.asr_model = self.load_en_asr_model()

    def load_en_asr_model(self):
        model = whisper.load_model("large-v3").to(self.device)
        return model

    def asr(self, audio_path):
        if self.dataset.lang == "en":
            transcription = self.asr_model.transcribe(audio_path)["text"]
        return transcription.strip()
        
    def evaluate(self, output_dir, rank=0, world_size=1):
        if not self.thinking:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(str(output_dir) + "_thinking")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        predictions_file_path_rank = output_dir / f"predictions_rank_{rank}.json"
        
        predictions_this_rank = []
        processed_item_ids_this_rank = set()
        
        if predictions_file_path_rank.exists():
            try:
                with open(predictions_file_path_rank, "r", encoding="utf-8") as f:
                    predictions_this_rank = json.load(f)
                for pred_item in predictions_this_rank:
                    processed_item_ids_this_rank.add(pred_item["uid"])
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
                item['uid'] = item_id
                items_to_process_this_run.append(item)
        
        if rank == 0:
            print(f"\n========== Running Multi-Challenge Evaluation ==========\n")
        
        progress_bar = tqdm.tqdm(items_to_process_this_run, desc=f"Running Multi-Challenge Evaluation (Rank {rank})", disable=(rank != 0))

        for item in progress_bar:
            print(f"\n----- Sample {item['uid']+1} (Rank {rank}) -----")
            assistant_voice_id = item['assistant_voice_id']
            prompt_speech_path = os.path.join(self.prompt_root_path, assistant_voice_id + ".wav")

            if self.task_type == 's2s':
                try:
                    speech_dialogue = item['speech_dialogue']
                    text_dialogue = item["CONVERSATION"]
                    output_audio_path = str(output_dir / f"audio_{item['uid']}.wav")
                    text_response = self.model.spoken_dialogue_sft_multiturn(speech_dialogue, text_dialogue, output_audio_path=output_audio_path, prompt_speech=prompt_speech_path)
                    text_response = text_response.strip().split('<|eot|>')[0].replace(".....", "").replace("...", "").replace("..", "")
                except Exception as e:
                    print(e)
                    text_response = ""
                    output_audio_path = ""
                    print('exception in s2s dialogue')
            elif self.task_type == 't2t':
                text_dialogue = item["CONVERSATION"]
                text_response = self.model.text_dialogue_sft_multiturn(text_dialogue)
                text_response = text_response.strip().split('<|eot|>')[0].replace(".....", "").replace("...", "").replace("..", "")
                output_audio_path = None
            
            elif self.task_type == 's2t':
                speech_dialogue = item['speech_dialogue']
                text_dialogue = item["CONVERSATION"]
                text_response = self.model.speech2text_dialogue_sft_multiturn(speech_dialogue, text_dialogue)
                text_response = text_response.strip().split('<|eot|>')[0].replace(".....", "").replace("...", "").replace("..", "")
                output_audio_path = None

            result_dict = {k: v for k, v in item.items() if k != 'audio'}
            result_dict['response'] = text_response
            result_dict["speech_response"] = output_audio_path

            if output_audio_path is not None and os.path.exists(output_audio_path):
                result_dict['asr_response'] = self.asr(output_audio_path)
            else:
                result_dict['asr_response'] = ""

            print("----------Text Response: ", text_response)
            print("----------Speech Response: ", output_audio_path)
            print("-" * 70)

            predictions_this_rank.append(result_dict)
            
            temp_file_path_rank = output_dir / f"predictions_rank_{rank}.json.tmp"
            with open(temp_file_path_rank, "w", encoding="utf-8") as f_tmp:
                json.dump(predictions_this_rank, f_tmp, indent=4, ensure_ascii=False)
            os.replace(temp_file_path_rank, predictions_file_path_rank)
        
        print(f"Rank {rank} Multi-Challenge Evaluation complete. Processed {len(items_to_process_this_run)} new items. Results saved to {predictions_file_path_rank}")

    def calculate_metrics(self, output_dir, rank=0, world_size=1):
        if rank != 0:
            return None
        
        if not self.thinking:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(output_dir + "_thinking")
        all_predictions = []
        
        temp_files = []
        pattern = f"predictions_rank_*.json"
        import glob
        
        temp_file_pattern = str(output_dir / pattern)
        found_temp_files = glob.glob(temp_file_pattern)
        
        if not found_temp_files:
            print(f"Warning: No temporary files found in {output_dir} (pattern: {pattern})")
            return None, []
        
        found_temp_files.sort(key=lambda x: int(Path(x).stem.split('_rank_')[-1]))
        
        for temp_file_path_str in found_temp_files:
            temp_file_path = Path(temp_file_path_str)
            temp_files.append(temp_file_path)
            
            rank_num = temp_file_path.stem.split('_rank_')[-1]
            
            if temp_file_path.exists():
                with open(temp_file_path, "r", encoding="utf-8") as f:
                    all_predictions.extend(json.load(f))
                print(f"Collected rank {rank_num} results: {temp_file_path}")
            else:
                print(f"Warning: File not found: {temp_file_path}")

        predictions_file_path = output_dir / "results.json"
        with open(predictions_file_path, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, indent=4, ensure_ascii=False)
        
        print(f"\n========== Multi-Challenge Evaluation Summary ==========\n")    
        print(f"Total items processed: {len(all_predictions)}")
        print(f"Results saved to: {predictions_file_path}")

        results_dict = {}
        total_score = 0
        if 'asr_response' in all_predictions[0] and all_predictions[0]['asr_response'] is not None:
            for i in range(len(all_predictions)):
                response = all_predictions[i]['asr_response']
                TARGET_QUESTION = all_predictions[i]['TARGET_QUESTION']
                PASS_CRITERIA = all_predictions[i]['PASS_CRITERIA']
                prompt = JUDGE_PROMPT.format(response, TARGET_QUESTION)
                print("----------{}---Prompt: ".format(i), prompt)
                try:
                    judgement = self.evaluation_model.generate(prompt)
                    all_predictions[i]['asr_verdict'] = judgement.verdict
                    if judgement.verdict == PASS_CRITERIA and response != "":
                        all_predictions[i]['asr_score'] = 1
                    else:
                        all_predictions[i]['asr_score'] = 0
                    all_predictions[i]['asr_reasoning'] = judgement.reasoning
                except Exception as e:
                    print(e)
                    all_predictions[i]['asr_verdict'] = ''
                    all_predictions[i]['asr_score'] = 0
                    all_predictions[i]['asr_reasoning'] = 'Error during evaluation'
                total_score += all_predictions[i]['asr_score']
            results_dict['ASR average_score'] = total_score / len(all_predictions) * 100

        if 'response' in all_predictions[0] and all_predictions[0]['response'] is not None:
            total_score = 0
            for i in range(len(all_predictions)):
                response = all_predictions[i]['response']
                TARGET_QUESTION = all_predictions[i]['TARGET_QUESTION']
                PASS_CRITERIA = all_predictions[i]['PASS_CRITERIA']
                prompt = JUDGE_PROMPT.format(response, TARGET_QUESTION)
                print("----------{}---Prompt: ".format(i), prompt)
                try:
                    judgement = self.evaluation_model.generate(prompt)
                    all_predictions[i]['verdict'] = judgement.verdict
                    if judgement.verdict == PASS_CRITERIA and response != "":
                        all_predictions[i]['score'] = 1
                    else:
                        all_predictions[i]['score'] = 0
                    all_predictions[i]['reasoning'] = judgement.reasoning
                except Exception as e:
                    print(e)
                    all_predictions[i]['verdict'] = ''
                    all_predictions[i]['score'] = 0
                    all_predictions[i]['reasoning'] = 'Error during evaluation'
                total_score += all_predictions[i]['score']

            results_dict['Text average_score'] = total_score / len(all_predictions) * 100
        results_dict['ASR Results'] = all_predictions
        
        with open(output_dir / ("score_4o.json"), "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
        
        return results_dict