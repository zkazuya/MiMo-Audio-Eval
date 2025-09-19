model_path=$1
tokenizer_path=$2
model_name=$3
dataset=speechmmlu
split=all
task=speechmmlu_s2t
n_few_shots=5

output_dir="${model_path}/eval.pretrain.speechmmlu_s2t.${dataset}.${split}.${n_few_shots}shots"

NUM_GPUS=8

torchrun --nproc_per_node=$NUM_GPUS main.py \
    --model $model_name \
    --model-path $model_path \
    --tokenizer-path $tokenizer_path \
    --dataset $dataset \
    --split $split \
    --task $task \
    --n-few-shots $n_few_shots \
    --output-dir $output_dir \
    --distribute --exec-mode infer

torchrun --nproc_per_node=$NUM_GPUS main.py \
    --model $model_name \
    --model-path $model_path \
    --tokenizer-path $tokenizer_path \
    --dataset $dataset \
    --split $split \
    --task $task \
    --n-few-shots $n_few_shots \
    --output-dir $output_dir \
    --distribute --exec-mode calculate