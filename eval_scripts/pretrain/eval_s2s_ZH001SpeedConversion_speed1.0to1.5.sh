model_path=$1
tokenizer_path=$2
model_name=$3
task=general_s2s
dataset=general_s2s
split=ZH001SpeedConversion_speed1.0to1.5
n_few_shots=16

output_dir="${model_path}/eval.pretrain.${task}"

NUM_GPUS=8

torchrun --nproc_per_node=$NUM_GPUS main.py \
    --model $model_name \
    --model-path $model_path \
    --tokenizer-path $tokenizer_path \
    --dataset $dataset \
    --task $task \
    --n-few-shots $n_few_shots \
    --output-dir $output_dir \
    --split $split \
    --distribute --exec-mode infer