#!/bin/bash
# for metagl
seeds=("1" "2" "3" "4" "5")
if_calculate_difficult_levels=("0")  # if 1, then divide test samples according to difficulty level
missing_sample_ratios=("0.0") # "0.2" "0.4" "0.6" "0.8"
missing_choice_ratios=("0.0") #  "0.2" "0.4" "0.6" "0.8"
limit_times=("1000") # "1.3" "1.1" "0.9" "0.7" "0.5" "0.3"
losses=("cce") #  "bce"
encoders=("blip") #  "vilt" "standard"

for if_calculate_difficult_level in "${if_calculate_difficult_levels[@]}"; do
    for missing_sample_ratio in "${missing_sample_ratios[@]}"; do
         for missing_choice_ratio in "${missing_choice_ratios[@]}"; do
            for limit_time in "${limit_times[@]}"; do
                for loss in "${losses[@]}"; do
                     for encoder in "${encoders[@]}"; do
                         for seed in "${seeds[@]}"; do
                             CUDA_VISIBLE_DEVICES=0 python code/run_metagl.py --seed $seed --encoder $encoder --loss $loss --limit_time $limit_time --missing_choice_ratio $missing_choice_ratio --missing_sample_ratio $missing_sample_ratio --if_calculate_difficult_level $if_calculate_difficult_level
                         done
                    done
                done
             done
        done
     done
done
