model_path=$1
tokenizer_path=$2
model_name=$3
task=dynamic_superb_s2s
dataset=dynamic_superb_s2s
split=DynamicSuperb/SuperbSE_VoiceBankDEMAND-Test
n_few_shots=16

output_dir="${model_path}/eval.pretrain.${task}.denoise"

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
    --sample-rate 24000 \
    --distribute --exec-mode infer 