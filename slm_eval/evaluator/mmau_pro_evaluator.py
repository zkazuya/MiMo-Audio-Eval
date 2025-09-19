# Copyright 2025 Xiaomi Corporation.
import os
import json
import torch
import tqdm
from pathlib import Path
import multiprocessing
import numpy as np
from typing import Optional, List, Dict, Any
import time
import logging
import requests
import json
import re
from datasets import Audio
import pdb
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from nltk.tokenize import sent_tokenize
import nltk
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch.nn.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_qwen_model(device):
    """Load the Qwen 2.5 model"""
    print("Loading Qwen 2.5 model...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    print("Qwen 2.5 model loaded successfully!")
    return model, tokenizer

def create_evaluation_prompt(question, reference_answer, model_response, task_type):
    """Create a structured prompt for the LLM judge to evaluate responses"""
    
    task_context = {
        "sound": "audio content analysis and sound identification",
        "speech": "speech recognition and conversation understanding", 
        "music": "music analysis and musical element identification",
        "open": "general open-ended question answering"
    }
    
    context = task_context.get(task_type, "general question answering")
    
    prompt = f"""You are an expert evaluator for {context} tasks. Please evaluate the quality of a model's response to a question.

Question: {question}

Reference Answer: {reference_answer}

Model Response: {model_response}

Please evaluate the model response on the following criteria and provide scores from 1-5 (where 5 is best):

1. **Correctness**: How factually accurate is the response compared to the reference?
2. **Relevance**: How well does the response address the specific question asked?
3. **Completeness**: Does the response cover all important aspects mentioned in the reference?
4. **Clarity**: How clear and well-structured is the response?

For each criterion, provide:
- A score from 1-5
- A brief justification (1-2 sentences)

Format your response as:

CORRECTNESS: [score] - [justification]
RELEVANCE: [score] - [justification] 
COMPLETENESS: [score] - [justification]
CLARITY: [score] - [justification]
OVERALL: [average score] - [overall assessment]"""

    return prompt

def extract_scores_from_evaluation(evaluation_text):
    """Extract numerical scores from the LLM judge evaluation"""
    scores = {}
    
    # Define patterns to extract scores
    patterns = {
        'correctness': r'CORRECTNESS:\s*(\d+)',
        'relevance': r'RELEVANCE:\s*(\d+)', 
        'completeness': r'COMPLETENESS:\s*(\d+)',
        'clarity': r'CLARITY:\s*(\d+)',
        'overall': r'OVERALL:\s*(\d+(?:\.\d+)?)'
    }
    
    for criterion, pattern in patterns.items():
        match = re.search(pattern, evaluation_text, re.IGNORECASE)
        if match:
            scores[criterion] = float(match.group(1))
        else:
            # Fallback: assign neutral score if not found
            scores[criterion] = 3.0
    
    # Calculate overall if not found
    if 'overall' not in scores or scores['overall'] == 3.0:
        criteria_scores = [scores.get(k, 3.0) for k in ['correctness', 'relevance', 'completeness', 'clarity']]
        scores['overall'] = np.mean(criteria_scores)
    
    return scores

def evaluate_openended_with_qwen(model, tokenizer, questions, reference_answers, model_responses, task_types):
    """Evaluate open-ended responses using Qwen 2.5 as a judge"""
    all_scores = []
    detailed_evaluations = []
    
    print("Performing Qwen 2.5 LLM judge evaluation...")
    
    for i, (question, ref_answer, model_response, task_type) in tqdm.tqdm(enumerate(zip(questions, reference_answers, model_responses, task_types))):
        print(f"Evaluating open-ended question {i+1} of {len(questions)}")
        # Create evaluation prompt
        eval_prompt = create_evaluation_prompt(question, ref_answer, model_response, task_type)
        
        # Tokenize and generate evaluation
        messages = [
            {"role": "system", "content": "You are a helpful and objective evaluator."},
            {"role": "user", "content": eval_prompt}
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate evaluation
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        evaluation_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract scores
        scores = extract_scores_from_evaluation(evaluation_text)
        all_scores.append(scores)
        print(scores)
        detailed_evaluations.append({
            'question': question,
            'reference_answer': ref_answer,
            'model_response': model_response,
            'evaluation': evaluation_text,
            'scores': scores,
            'task_type': task_type
        })
    
    return all_scores, detailed_evaluations

def calculate_openended_metrics(all_scores):
    """Calculate evaluation metrics from LLM judge scores"""
    if not all_scores:
        return {}
    
    # Calculate averages for each criterion
    criteria = ['correctness', 'relevance', 'completeness', 'clarity', 'overall']
    metrics = {}
    
    for criterion in criteria:
        scores = [score_dict.get(criterion, 3.0) for score_dict in all_scores]
        metrics[f'avg_{criterion}'] = np.mean(scores)
        metrics[f'std_{criterion}'] = np.std(scores)
    
    # Calculate percentage of good responses (score >= 4.0)
    good_responses = sum(1 for scores in all_scores if scores.get('overall', 3.0) >= 4.0)
    metrics['good_response_rate'] = good_responses / len(all_scores)
    
    # Calculate percentage of poor responses (score <= 2.0)
    poor_responses = sum(1 for scores in all_scores if scores.get('overall', 3.0) <= 2.0)
    metrics['poor_response_rate'] = poor_responses / len(all_scores)
    
    return metrics

# ================================
# Audio Instruction Following (AIF) Evaluation Functions
# ================================

def count_words(text):
    return len(text.split())

def count_sentences(text):
    sentences = sent_tokenize(text)
    return len(sentences)

def count_paragraphs(text):
    # paragprah1 *** paragraph2
    paragraphs = text.split("***")
    return len([p for p in paragraphs if p.strip()])

def count_bullet_points(text):
    # Match bullets at the start of the string OR start of a line
    bullets = re.findall(r'(?:^|\n)\s*\*\s+', text)
    return len(bullets)

def count_highlighted_sections(text):
    # * highlight *
    highlights = re.findall(r'\*([^*]+)\*', text)
    return len(highlights)

def count_placeholders(text):
    # [placeholder]
    placeholders = re.findall(r'\[[^\]]+\]', text)
    return len(placeholders)

def count_capital_words(text):
    words = text.split()
    capital_words = []
    for word in words:
        if word.isupper():
            capital_words.append(word)
    return len(capital_words)

def count_keyword_frequency(text, keyword):
    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
    matches = re.findall(pattern, text.lower())
    return len(matches)

def has_title(text):
    # <<title>>
    return bool(re.search(r'<<[^>]+>>', text))

def has_postscript(text, marker):
    text_alpha = re.sub(r'[^a-zA-Z]', '', text).lower()
    marker_alpha = re.sub(r'[^a-zA-Z]', '', marker).lower()
    return marker_alpha in text_alpha

def starts_with_phrase(text, phrase):
    text_alpha = re.sub(r'[^a-zA-Z ]', '', text).lower()
    phrase_alpha = re.sub(r'[^a-zA-Z ]', '', phrase).lower()
    return text_alpha.startswith(phrase_alpha)

def ends_with_phrase(text, phrase):
    text_alpha = re.sub(r'[^a-zA-Z ]', '', text).lower()
    phrase_alpha = re.sub(r'[^a-zA-Z ]', '', phrase).lower()
    return text_alpha.endswith(phrase_alpha)

def is_wrapped_in_quotes(text):
    stripped = text.strip()
    return stripped.startswith('"') and stripped.endswith('"')

def has_no_commas(text):
    return ',' not in text

def check_sections(text, num_sections, splitter):
    # Escape the splitter in case it contains special regex characters
    escaped_splitter = re.escape(splitter)
    
    # Split on the exact delimiter (not inside words)
    sections = re.split(rf'\s*{escaped_splitter}\s*', text.strip())
    
    # Remove empty/whitespace-only sections
    actual_sections = [s for s in sections if s.strip()]
    
    return len(actual_sections) == num_sections

def evaluate_aif_sample(response, sample_data):
    """Evaluate Audio Instruction Following sample"""
    task_identifier = sample_data.get("task_identifier", "")
    kwargs = sample_data.get("kwargs", {}) or {}
    
    success = False
    
    if task_identifier == "Include Keywords":
        keywords = kwargs.get("keywords", "").split(", ")
        success = all(keyword.lower() in response.lower() for keyword in keywords)
        
    elif task_identifier == "Keyword Frequency":
        keyword = kwargs.get("keyword", "")
        target = kwargs.get("N", 0)
        actual = count_keyword_frequency(response, keyword)
        success = actual == target
    
    elif task_identifier == "Forbidden Words":
        forbidden_words = kwargs.get("forbidden_words", "").split(", ")
        success = not any(word.lower() in response.lower() for word in forbidden_words)
    
    elif task_identifier == "Number Paragraphs":
        target = kwargs.get("N", 0)
        actual = count_paragraphs(response)
        success = actual == target
    
    elif task_identifier == "Number Words (at least)":
        target = kwargs.get("N", 0)
        actual = count_words(response)
        success = actual >= target

    elif task_identifier == "Number Words (at most)":
        target = kwargs.get("N", 0)
        actual = count_words(response)
        success = actual <= target
        
    elif task_identifier == "Number Words (range)":
        N1 = kwargs.get("N1", 0)
        N2 = kwargs.get("N2", 999)
        actual = count_words(response)
        success = N1 <= actual <= N2
    
    elif task_identifier == "Number Sentences (at least)":
        target = kwargs.get("N", 0)
        actual = count_sentences(response)
        success = actual >= target
        
    elif task_identifier == "Number Sentences (at most)":
        target = kwargs.get("N", 0)
        actual = count_sentences(response)
        success = actual <= target
    
    elif task_identifier == "Number Sentences (range)":
        N1 = kwargs.get("N1", 0)
        N2 = kwargs.get("N2", 999)
        actual = count_sentences(response)
        success = N1 <= actual <= N2
        
    elif task_identifier == "Postscript":
        marker = kwargs.get("postscript_marker", "")
        success = has_postscript(response, marker)
    
    elif task_identifier == "Number Placeholder":
        target = kwargs.get("N", 0)
        actual = count_placeholders(response)
        success = actual >= target
    
    elif task_identifier == "Number Bullets":
        target = kwargs.get("N", 0)
        actual = count_bullet_points(response)
        success = actual == target
    
    elif task_identifier == "Title":
        success = has_title(response)
    
    elif task_identifier == "Minimum Number Highlighted Section":
        target = kwargs.get("N", 0)
        actual = count_highlighted_sections(response)
        success = actual >= target
    
    elif task_identifier == "Multiple Sections":
        target = kwargs.get("N", 0)
        splitter = kwargs.get("section_splitter", "")
        success = check_sections(response, target, splitter)
    
    elif task_identifier == "Repeat Prompt":
        original_prompt = sample_data.get("prompt_transcription", "")
        success = response.strip().lower().startswith(original_prompt.strip().lower())
    
    elif task_identifier == "Two Responses":
        separator = "******"
        parts = response.split(separator)
        success = len(parts) == 2 and parts[0].lower().strip() != parts[1].lower().strip()
    
    elif task_identifier == "All Uppercase":
        success = response.isupper()
    
    elif task_identifier == "All Lowercase":
        success = response.islower()
    
    elif task_identifier == "All-capital Words (at least)":
        target = kwargs.get("N", 0)
        actual = count_capital_words(response)
        success = actual >= target
    
    elif task_identifier == "All-capital Words (at most)":
        target = kwargs.get("N", 0)
        actual = count_capital_words(response)
        success = actual <= target
    
    elif task_identifier == "All-capital Words (range)":
        N1 = kwargs.get("N1", 0)
        N2 = kwargs.get("N2", 999)
        actual = count_capital_words(response)
        success = N1 <= actual <= N2
    
    elif task_identifier == "Start Checker":
        phrase = kwargs.get("start_phrase", "")
        success = starts_with_phrase(response, phrase)
    
    elif task_identifier == "End Checker":
        phrase = kwargs.get("end_phrase", "")
        success = ends_with_phrase(response, phrase)
    
    elif task_identifier == "Quotation":
        success = is_wrapped_in_quotes(response)
    
    elif task_identifier == "No Commas":
        success = has_no_commas(response)
    
    return success


# ================================
# Closed-ended Evaluation Functions (NVEmbed)
# ================================

def load_nvembed_model(device):
    """Load the NVEmbed model"""
    print("Loading NV-Embed-v2 model...")
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True, local_files_only=False)
    model.to(device)
    print("NVEmbed model loaded successfully!")
    return model


def evaluate_closedended_with_nvembed(model, questions, choices_list, ground_truth_answers, predicted_answers, task_types):
    """NVEmbed-based evaluation: match predictions to choices using embedding similarity"""
    predictions = []
    confidence_scores = []
    
    print("Performing NVEmbed evaluation...")
    
    for i, (question, choices, gt_answer, model_prediction, task_type) in tqdm.tqdm(enumerate(zip(questions, choices_list, ground_truth_answers, predicted_answers, task_types))):
        pdb.set_trace()
        # Encode the prediction (what the model said)
        prediction_embedding = model.encode([model_prediction], instruction="", max_length=4096)
        prediction_embedding = F.normalize(prediction_embedding, p=2, dim=1)
        
        # Encode each choice option
        choice_embeddings = model.encode(choices, instruction="", max_length=4096)
        choice_embeddings = F.normalize(choice_embeddings, p=2, dim=1)
        
        # Calculate similarity between prediction and each choice
        scores = (prediction_embedding @ choice_embeddings.T) * 100
        scores = scores.squeeze()
        
        # Find the choice most similar to the prediction
        best_choice_idx = torch.argmax(scores).item()
        matched_choice = choices[best_choice_idx]
        confidence = torch.max(scores).item()
        
        predictions.append(matched_choice)
        confidence_scores.append(confidence)
    
    return predictions, confidence_scores

# ================================
# Utility Functions
# ================================

def calculate_metrics(ground_truth, predictions):
    """Calculate evaluation metrics"""
    if len(ground_truth) == 0 or len(predictions) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics

def calculate_weighted_performance(category_results):
    """Calculate overall weighted performance using preferred metrics for each type"""
    total_weighted_score = 0.0
    total_samples = 0
    category_scores = {}
    
    for category, result in category_results.items():
        count = result['count']
        
        # Use preferred performance metric for each type
        if result['type'] == 'openended':
            # Use avg_overall score (1-5 scale), normalize to 0-1 for weighting
            score = result['metrics'].get('avg_overall', 3.0) / 5.0
        elif result['type'] == 'aif':
            # Use success rate directly (0-1 scale)
            score = result['success_rate']
        elif result['type'] == 'closed':
            # Use accuracy as preferred metric (0-1 scale)
            score = result['metrics'].get('accuracy', 0.0)
        else:
            # Unknown type, skip
            continue
        
        category_scores[category] = score
        total_weighted_score += score * count
        total_samples += count
    
    overall_weighted_performance = total_weighted_score / total_samples if total_samples > 0 else 0.0
    
    return overall_weighted_performance, category_scores

class MMAUProEvaluator:

    def __init__(self, model, dataset, device=None, model_type='base', n_few_shots=0, thinking=False):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model_type = model_type
        self.n_few_shots = n_few_shots
        self.thinking = thinking
    def format_qa(self, question, choices):
        final_output = question + '\n\n' + 'Choice: \n'
        for i in range(len(choices)):
            final_output += f"{choices[i]}\n"
        final_output += f"\n"
        
        final_output += f'Make a choice from the given {len(choices)} choices.'

        return final_output
    def string_match(self, answer, prediction, choices):
        # Function to normalize and tokenize text
        def tokenize(text):
            # Convert to lowercase and find all word tokens
            return set(re.findall(r'\b\w+\b', text.lower()))
        
        # Tokenize prediction and answer
        prediction_tokens = tokenize(prediction)
        answer_tokens = tokenize(answer)
        
        if not prediction_tokens:
            return False
        
        # Tokenize incorrect choices and exclude tokens present in the answer
        incorrect_tokens = set()
        for choice in choices:
            choice_tokens = tokenize(choice)
            if choice_tokens != answer_tokens:
                incorrect_tokens.update(choice_tokens - answer_tokens)
        
        # Condition 1: All tokens of the answer are in the prediction
        cond1 = answer_tokens.issubset(prediction_tokens)
        
        # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
        cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
        
        return cond1 and cond2

    def format_instruct_qa(self, question, choices, audio, category):
        if category == 'open' or category == 'instruction following':
            final_output = "<sound>" + question + '\n'
            audio = audio[0]
        elif category == 'multi':
            final_output = 'Given the following multiple audios, and answer the question.'
            for i in range(len(audio)):
                final_output += f'This is the {i+1}th audio: <sound>\n'
            final_output += question + '\n\n' + 'Choice: \n'
            for i in range(len(choices)):
                final_output += f"{choices[i]}\n"
            final_output += f"\n"
            final_output += f'Choose a choices from the given {len(choices)} choices. Do not provide any additional explanations or content. Output must match exactly one of the listed choices.'
        else:
            final_output = "<sound>" + question + '\n\n' + 'Choice: \n'
            for i in range(len(choices)):
                final_output += f"{choices[i]}\n"
            final_output += f"\n"

            final_output += f'Choose a choices from the given {len(choices)} choices. Do not provide any additional explanations or content. Output must match exactly one of the listed choices.'
            audio = audio[0]

        instruction = [
            {
                "from": "human",
                "value": [
                    {
                        "type": "text",
                        "value": final_output
                    },
                    {
                        "type": "sound",
                        "value": audio
                    }
                ]
            }
        ]
        return instruction

    def few_shot_qa(self, question, choices, audio, audio_path, past_examples):
        final_output = []
        for i in range(len(past_examples)):
            final_output.append({
                'question': self.format_qa(past_examples[i]['question'], past_examples[i]['choices'].split('|')),
                'audio': torch.tensor(past_examples[i]['audio']["array"]).unsqueeze(0).float(),
                'answer': past_examples[i]['answer'],
                'audio_path': past_examples[i]['audio_path']
            })
        final_output.append({
            'question': self.format_qa(question, choices),
            'audio': audio,
            'audio_path': audio_path
        })
        
        return final_output

    def get_few_shot_exclude(self, dataset, n_few_shots, exclude_index):
        import random
        
        if not 0 <= exclude_index < len(dataset):
            raise IndexError(f"exclude_index ({exclude_index}) 超出范围 (0~{len(dataset)-1})")
        
        candidates = dataset[:exclude_index] + dataset[exclude_index+1:]
        
        if n_few_shots > len(candidates):
            raise ValueError(f"n_few_shots ({n_few_shots}) 不能大于候选样本数 ({len(candidates)})")
        
        return random.sample(candidates, n_few_shots)
        
    def evaluate(self, output_dir, rank=0, world_size=1):
        set_seed(42)
        if not self.thinking:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(str(output_dir) + "_thinking_v2")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        predictions_file_path_rank = output_dir / f"predictions_rank_{rank}.json"
        
        predictions_this_rank = []
        processed_item_ids_this_rank = set()
        
        # Load checkpoint if exists
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
        
        # Shard dataset for multi-GPU evaluation
        dataset_shard = list(self.dataset)[rank::world_size]
        items_to_process_this_run = []

        for idx, item in enumerate(dataset_shard):
            # Create item with proper ID for sharded processing
            item_id = rank + idx * world_size
            if item_id not in processed_item_ids_this_rank:
                item['uid'] = item_id
                items_to_process_this_run.append(item)
        
        if rank == 0:
            print(f"\n========== Running MMAU-Pro Evaluation ==========\n")
        
        progress_bar = tqdm.tqdm(items_to_process_this_run, desc=f"Running MMAU-Pro Evaluation (Rank {rank})", disable=(rank != 0))

        index = 0
        for item in progress_bar:
            print(f"\n----- Sample {item['uid']+1} (Rank {rank}) -----")

            audio = item['audio_path']
            question = item['question']
            audio_path = item['audio_path']
            choices = item['choices']
            
            if self.model_type == 'base':
                if self.n_few_shots > 0:
                    response = self.model.few_shots_qa(self.few_shot_qa(question, choices, audio, audio_path, self.get_few_shot_exclude(items_to_process_this_run, self.n_few_shots, index)))
                else:
                    response = self.model.qa(audio, self.format_qa(question, choices))
            elif self.model_type == 'instruct':
                response = self.model.instruction_following(self.format_instruct_qa(question, choices, audio, item['category']), append_generation_prompt=True, thinking=self.thinking)
            
            result_dict = {k: v for k, v in item.items() if k != 'audio'}
            result_dict['response'] = response
            print("++++++++++++++++++++++++++ Question ++++++++++++++++++++++++++")
            print("Question: ", item['question'])
            print("++++++++++++++++++++++++++ choices ++++++++++++++++++++++++++")
            print("Choices: ", choices)
            print("++++++++++++++++++++++++++ Response ++++++++++++++++++++++++++")
            print("Response: ", response)
            print("-" * 70)
            
            predictions_this_rank.append(result_dict)
            
            # Save results incrementally using temporary file for atomic operation
            temp_file_path_rank = output_dir / f"predictions_rank_{rank}.json.tmp"
            with open(temp_file_path_rank, "w", encoding="utf-8") as f_tmp:
                json.dump(predictions_this_rank, f_tmp, indent=4, ensure_ascii=False)
            os.replace(temp_file_path_rank, predictions_file_path_rank)
            index += 1
        
        print(f"Rank {rank} MMAU-Pro evaluation complete. Processed {len(items_to_process_this_run)} new items. Results saved to {predictions_file_path_rank}")

    def calculate_acc(self, all_predictions):
        # Initialize results storage
        all_results = {}
        category_results = {}
        print("\n" + "="*60)
        print("EVALUATING OPEN-ENDED QUESTIONS")
        print("="*60)

        open_df = [item for item in all_predictions if item['category'] == 'open']
        if len(open_df) > 0:
            print(f"Found {len(open_df)} open-ended questions")

            qwen_model, qwen_tokenizer = load_qwen_model(self.device)
            questions = [open_item['question'] for open_item in open_df]
            reference_answers = [open_item['answer'] for open_item in open_df]
            model_responses = [open_item['response'] for open_item in open_df]
            task_types = ['open'] * len(open_df)

            openended_scores, openended_detailed = evaluate_openended_with_qwen(
                qwen_model, qwen_tokenizer, questions, reference_answers, model_responses, task_types
            )

            # Calculate metrics
            openended_metrics = calculate_openended_metrics(openended_scores)
            category_results['open'] = {
                'type': 'openended',
                'count': len(open_df),
                'metrics': openended_metrics,
                'scores': openended_scores
            }
            
            print(f"Open-ended evaluation completed: {len(openended_scores)} samples")
            print(f"Average Overall Score: {openended_metrics.get('avg_overall', 0.0):.3f}/5.0")
            
            # Clean up memory
            del qwen_model
            del qwen_tokenizer
            torch.cuda.empty_cache()
        else:
            print("No open-ended questions found")
            openended_scores = []


            all_results['open'] = open_df
            category_results['open'] = open_df
        
        print("\n" + "="*60)
        print("EVALUATING INSTRUCTION FOLLOWING QUESTIONS")
        print("="*60)
        
        # ================================
        # Process Instruction Following Questions
        # ================================
        print("\n" + "="*60)
        print("EVALUATING INSTRUCTION FOLLOWING QUESTIONS")
        print("="*60)

        aif_df = [item for item in all_predictions if item['category'] == 'instruction following']
        
        if len(aif_df) > 0:
            print(f"Found {len(aif_df)} instruction following questions")
            
            aif_results = []
            for idx, item in enumerate(aif_df):
                model_response = item['response']
                sample_data = {
                    'task_identifier': item['task_identifier'],
                    'kwargs': item['kwargs'],
                    'prompt_transcription': item['question'] if 'question' in item else ""
                }
                success = evaluate_aif_sample(model_response, sample_data)
                aif_results.append(success)
            
            # Calculate AIF metrics
            success_rate = np.mean([float(r) for r in aif_results])
            category_results['instruction following'] = {
                'type': 'aif',
                'count': len(aif_df),
                'success_rate': success_rate,
                'results': aif_results
            }
            
            print(f"Instruction following evaluation completed: {len(aif_results)} samples")
            print(f"Success Rate: {success_rate:.3f}")
        else:
            print("No instruction following questions found")
            aif_results = []

        # ================================
        # Process Closed-ended Questions  
        # ================================
        print("\n" + "="*60)
        print("EVALUATING CLOSED-ENDED QUESTIONS")
        print("="*60)
        
        closed_categories = set([item['category'] for item in all_predictions if item['category'] not in ['open', 'instruction following']])
        closed_categories = list(closed_categories)

        closed_df = [item for item in all_predictions if item['category'] not in ['open', 'instruction following']]
        
        if len(closed_df) > 0:
            print(f"Found {len(closed_df)} closed-ended questions across categories: {closed_categories}")
            
            # Filter out samples without proper choices
            closed_df = [item for item in closed_df if item['choices'] is not None]
            # closed_df = [item for item in closed_df if item['choices'].notna()]
            closed_df = [item for item in closed_df if len(item['choices']) > 1]
            # closed_df = closed_df[closed_df['choices'].notna()].copy()
            # closed_df = closed_df[closed_df['choices'].apply(lambda x: len(x) > 1 if hasattr(x, '__len__') else False)].copy()
            
            print(f"After filtering: {len(closed_df)} samples with valid choices")
            pdb.set_trace()
            if len(closed_df) > 0:
                # Load NVEmbed model for closed-ended evaluation
                nvembed_model = load_nvembed_model(self.device)
                
                questions = [closed_item['question'] for closed_item in closed_df]
                ground_truth_answers = [closed_item['answer'] for closed_item in closed_df]
                choices_list = [list(closed_item['choices']) if hasattr(closed_item['choices'], '__iter__') else [str(closed_item['choices'])] 
                            for closed_item in closed_df]
                model_predictions = [closed_item['response'] for closed_item in closed_df]
                task_types = [closed_item['category'] for closed_item in closed_df]
                # questions = closed_df['question'].tolist()
                # ground_truth_answers = closed_df['answer'].tolist()
                # choices_list = [list(choices) if hasattr(choices, '__iter__') else [str(choices)] 
                #             for choices in closed_df['choices'].tolist()]
                # model_predictions = closed_df[model_output_column].fillna("").tolist()
                # task_types = closed_df['category'].tolist()
                
                # Evaluate closed-ended questions
                predictions, confidence_scores = evaluate_closedended_with_nvembed(
                    nvembed_model, questions, choices_list, ground_truth_answers, model_predictions, task_types
                )
                
                # Calculate metrics overall and by category
                overall_closed_metrics = calculate_metrics(ground_truth_answers, predictions)
                
                # Store results by category
                for category in closed_categories:
                    cat_mask = closed_df['category'] == category
                    if cat_mask.sum() > 0:
                        cat_gt = [ground_truth_answers[i] for i, mask in enumerate(cat_mask) if mask]
                        cat_pred = [predictions[i] for i, mask in enumerate(cat_mask) if mask]
                        cat_metrics = calculate_metrics(cat_gt, cat_pred)
                        
                        category_results[category] = {
                            'type': 'closed',
                            'count': cat_mask.sum(),
                            'metrics': cat_metrics
                        }
                
                print(f"Closed-ended evaluation completed: {len(predictions)} samples")
                print(f"Overall Accuracy: {overall_closed_metrics['accuracy']:.4f}")
                
                # Clean up memory
                del nvembed_model
                torch.cuda.empty_cache()
            else:
                print("No valid closed-ended questions with choices found")
                predictions = []
                overall_closed_metrics = {}
        else:
            print("No closed-ended questions found")
            predictions = []
            overall_closed_metrics = {}

        # ================================
        # Calculate Weighted Performance
        # ================================
        # overall_weighted_performance, category_scores = calculate_weighted_performance(category_results)
        
        # ================================
        # Generate Comprehensive Report
        # ================================
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        
        total_samples = len(all_predictions)
        evaluated_samples = sum([result['count'] for result in category_results.values()])
        
        print(f"Total samples: {total_samples}")
        print(f"Successfully evaluated: {evaluated_samples}")
        print(f"Overall Weighted Performance: {overall_weighted_performance:.4f}")
        
        print("\nBREAKDOWN BY CATEGORY:")
        print("-" * 60)
        
        for category, result in category_results.items():
            print(f"\n{category.upper()}:")
            print(f"  Type: {result['type']}")
            print(f"  Count: {result['count']}")
            print(f"  Performance Score: {category_scores.get(category, 0.0):.4f}")
            
            if result['type'] == 'openended':
                metrics = result['metrics']
                print(f"  Avg Overall Score: {metrics.get('avg_overall', 0.0):.3f}/5.0")
                print(f"  Avg Correctness: {metrics.get('avg_correctness', 0.0):.3f}/5.0")
                
            elif result['type'] == 'aif':
                print(f"  Success Rate: {result['success_rate']:.3f}")
                
            elif result['type'] == 'closed':
                metrics = result['metrics']
                print(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
                print(f"  F1 Score: {metrics.get('f1_score', 0.0):.4f}")
        
        # ================================
        # Save Results
        # ================================
        results_summary = {
            'evaluation_summary': {
                'total_samples': total_samples,
                'evaluated_samples': evaluated_samples,
                'overall_weighted_performance': overall_weighted_performance
            },
            'category_results': {}
        }
        
        # Format results for JSON serialization
        for category, result in category_results.items():
            json_result = {
                'type': result['type'],
                'count': result['count'],
                'performance_score': category_scores.get(category, 0.0)
            }
            
            if result['type'] == 'openended':
                json_result['metrics'] = result['metrics']
                
            elif result['type'] == 'aif':
                json_result['success_rate'] = result['success_rate']
                
            elif result['type'] == 'closed':
                json_result['metrics'] = result['metrics']
            
            results_summary['category_results'][category] = json_result

        print("\nEvaluation completed successfully!")
        print(f"\n🎯 SUMMARY:")
        print(f"   Overall Weighted Performance: {overall_weighted_performance:.4f}")
        print(f"   Total Samples Evaluated: {evaluated_samples}/{total_samples}")
        print("\nNOTE: This comprehensive evaluation requires:")
        print("- Qwen 2.5 LLM judge for open-ended questions (no fallback)")
        print("- Format constraint checking for instruction following") 
        print("- NVEmbed similarity matching for closed-ended questions (no fallback)")
        print("\nPerformance metric per category:")
        print("- Open-ended: LLM overall score / 5.0 (continuous quality assessment)")
        print("- Instruction following: Success rate (binary constraint satisfaction)")
        print("- Closed-ended: Classification accuracy (exact match rate)")
        print("- Overall metric is weighted by sample count per category")

        return results_summary
    
    def calculate_metrics(self, output_dir, rank=0, world_size=1):
        """Aggregate results from all ranks and calculate metrics using MMAU evaluators"""
        if rank != 0:
            return None
        
        if not self.thinking:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(output_dir + "_thinking_v2")
        all_predictions = []
        
        # Collect predictions from all ranks
        for r in range(world_size):
            predictions_file_path_rank = output_dir / f"predictions_rank_{r}.json"
            if predictions_file_path_rank.exists():
                with open(predictions_file_path_rank, "r", encoding="utf-8") as f:
                    all_predictions.extend(json.load(f))
            else:
                print(f"Warning: Predictions file not found for rank {r} at {predictions_file_path_rank}")
        
        # Save aggregated results
        predictions_file_path = output_dir / "results.json"
        with open(predictions_file_path, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, indent=4, ensure_ascii=False)
        
        print(f"\n========== MMAU-Pro Evaluation Summary ==========\n")

        print(f"Total items processed: {len(all_predictions)}")
        print(f"Results saved to: {predictions_file_path}")

        results_summary = self.calculate_acc(all_predictions)
        
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=4, ensure_ascii=False)

        return results_summary