model_path=$1
tokenizer_path=$2
model_name=$3
task=voice_conversion_s2s
dataset=esd
n_few_shots=16

output_dir="${model_path}/eval.pretrain.${task}.${dataset}"

NUM_GPUS=8

torchrun --nproc_per_node=$NUM_GPUS main.py \
    --model $model_name \
    --model-path $model_path \
    --tokenizer-path $tokenizer_path \
    --dataset $dataset \
    --task $task \
    --n-few-shots $n_few_shots \
    --output-dir $output_dir \
    --distribute --exec-mode infer 

python main.py \
    --model $model_name \
    --model-path $model_path \
    --tokenizer-path $tokenizer_path \
    --dataset $dataset \
    --task $task \
    --n-few-shots $n_few_shots \
    --output-dir $output_dir \
    --exec-mode calculate