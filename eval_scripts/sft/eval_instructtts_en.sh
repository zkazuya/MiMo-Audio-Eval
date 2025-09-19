model_path=$1
tokenizer_path=$2
model_name=$3
dataset=instruct_tts
task=instruct_tts
split=en

output_dir="${model_path}/eval.sft.${task}.${dataset}.${split}"

NUM_GPUS=8

torchrun --nproc_per_node=$NUM_GPUS main.py \
    --model $model_name \
    --model-path $model_path \
    --model-type instruct \
    --tokenizer-path $tokenizer_path \
    --dataset $dataset \
    --split $split \
    --task $task \
    --output-dir $output_dir \
    --distribute --exec-mode infer

torchrun --nproc_per_node=$NUM_GPUS main.py \
    --model $model_name \
    --model-path $model_path \
    --model-type instruct \
    --tokenizer-path $tokenizer_path \
    --dataset $dataset \
    --split $split \
    --task $task \
    --output-dir $output_dir \
    --distribute --exec-mode calculate