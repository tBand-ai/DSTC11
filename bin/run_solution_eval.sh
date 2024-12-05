#!/bin/bash

# Prepare directories for intermediate results of each subtask
eval_dataset=val
mkdir -p pred/${eval_dataset}

# eval for turn detection
model_name=microsoft/deberta-v3-base
model_name_exp=deberta-v3-base
checkpoint=runs/td-review-${model_name_exp}-solution
td_output_file=pred/${eval_dataset}/solution.td.${model_name_exp}.json
cuda_id=3

echo -e "\n@1@\n\n"

#CUDA_VISIBLE_DEVICES=${cuda_id} python solution.py \
python solution.py \
        --task detection \
        --eval_only \
        --model_name_or_path ${model_name} \
        --checkpoint ${checkpoint} \
        --dataroot data \
        --eval_dataset ${eval_dataset} \
	--no_labels \
        --knowledge_file knowledge.json \
        --output_file ${td_output_file}

echo -e "\n@2@\n\n"

# track entities
em_output_file=pred/${eval_dataset}/solution.em.${model_name_exp}.json
python solution/entity_matching.py \
        --dataroot data \
        --eval_dataset ${eval_dataset} \
        --labels_file ${td_output_file} \
        --output_file ${em_output_file}

echo -e "\n@3@\n\n"

# eval for knowledge selection
model_name=microsoft/deberta-v3-base
model_name_exp=deberta-v3-base
checkpoint=runs/ks-review-${model_name_exp}-oracle-solution
ks_output_file=pred/${eval_dataset}/solution.ks.${model_name_exp}.json
cuda_id=3

#CUDA_VISIBLE_DEVICES=${cuda_id} python3 solution.py \
python solution.py \
        --eval_only \
        --task selection \
        --model_name_or_path ${model_name_exp} \
        --checkpoint ${checkpoint} \
        --dataroot data \
        --eval_dataset ${eval_dataset} \
        --labels_file ${em_output_file} \
        --output_file ${ks_output_file} \
        --knowledge_file knowledge.json

echo -e "\n@4@\n\n"

# test for response generation
model_name=facebook/bart-base
model_name_exp=bart-base
checkpoint=runs/rg-review-${model_name_exp}-oracle-solution
rg_output_file=pred/${eval_dataset}/solution.rg.${model_name_exp}.json
cuda_id=3

#CUDA_VISIBLE_DEVICES=${cuda_id} python3 solution.py \
python solution.py \
        --task generation \
        --generate runs/rg-review-${model_name_exp}-solution \
        --generation_params_file solution/configs/generation/generation_params.json \
        --eval_dataset ${eval_dataset} \
        --dataroot data \
        --labels_file ${ks_output_file} \
        --knowledge_file knowledge.json \
        --output_file ${rg_output_file}

echo -e "\n@5@\n\n"

# verify and evaluate the output
rg_output_score_file=pred/${eval_dataset}/solution.rg.${model_name_exp}.score.json
python -m scripts.check_results --dataset ${eval_dataset} --dataroot data --outfile ${rg_output_file}
python -m scripts.scores --dataset ${eval_dataset} --dataroot data --outfile ${rg_output_file} --scorefile ${rg_output_score_file}
