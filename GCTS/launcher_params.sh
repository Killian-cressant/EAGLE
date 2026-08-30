#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

#FOR TELCO DATASET FIRST
learning_rate=(1e-3)
window_size=(20)
temporal_window=(3)
top_k=(3 5 10)
#top_k_centrality=(3 5)
output_dim=(6 8)
hidden_dim=(12 16)
num_layers=(3) #5
pos_weight_exponent=(1 3)
best_acc=0
best_args=""

for lr in "${learning_rate[@]}"; do
for ws in "${window_size[@]}"; do
for tw in "${temporal_window[@]}"; do
for k in "${top_k[@]}"; do
#for kc in "${top_k_centrality[@]}"; do
for o in "${output_dim[@]}"; do
for hd in "${hidden_dim[@]}"; do
for nl in "${num_layers[@]}"; do
for pwe in "${pos_weight_exponent[@]}"; do

    echo "Running: lr=$lr, ws=$ws, tw=$tw, top_k=$k, top_k_centrality=$kc, output_dim=$o, hidden_dim=$hd, num_layers=$nl, pos_weight_exp=$pwe"
    output=$(python3.12 run_GCTS.py \
        --learning_rate "$lr" \
        --window_size "$ws" \
        --max_epochs "100" \
        --horizon "$ws" \
        --temporal_window "$tw" \
        --output_dim "$o" \
        --hidden_dim "$hd" \
        --top_k "$k" \
        --num_layers "$nl" \
        --pos_weight_exponent "$pwe" \
        --experiment_name "FTS" \
        --dataset "telco" \
        --graph_type "corr" \
        --device "0" \
        2>&1) #--top_k_centrality "$kc" \

    echo "$output"
    acc=$(echo "$output" | grep -oP 'test_accuracy:\s*\K[0-9]+(\.[0-9]+)?')

    if (( $(echo "$acc > $best_acc" | bc -l) )); then
        best_acc=$acc
        best_args="lr=$lr ws=$ws tw=$tw top_k=$k top_k_centrality=$kc output_dim=$o hidden_dim=$hd num_layers=$nl pos_weight_exp=$pwe"
    fi
done
done
done
done
done
done
done
done
#done

echo "==============================="
echo "Best accuracy: $best_acc"
echo "Best args: $best_args"