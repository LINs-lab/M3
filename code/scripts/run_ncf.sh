#!/bin/bash
hidden_sizes="32"
lrs="0.01"
batchs="64"
log_tag="debug_ncf"
encoders="blip"
seeds="1 2 3 4 5"
optimizer="adam"
weight_decays="0.00" 
missing_choices="0.0" #  0.0 0.2 0.4 0.6 0.8
missing_samples="0.0" # 0.0 0.2 0.4 0.6 0.8
gammas="0.6"
losses="cce" # "bce cce"

for loss in $losses; do
    for gamma in $gammas; do
        for m1 in $missing_choices; do
            for m2 in $missing_samples; do
                for weight_decay in $weight_decays; do
                    for lr in $lrs; do
                        for batch in $batchs; do
                            for encoder in $encoders; do
                                for hidden_size in $hidden_sizes; do
                                    for seed in $seeds; do
                                        python code/run_ncf.py --hidden_size $hidden_size --lr $lr --log_tag $log_tag --encoder $encoder --gamma $gamma \
                                        --batch_size $batch --seed $seed --optimizer $optimizer --weight_decay $weight_decay --missing_choice_ratio $m1 --missing_sample_ratio $m2\
                                        --loss $loss
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done