#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# FOR TELCO DATASET FIRST
learning_rate=(1e-3 1e-4)
window_size=(20)
temporal_window=(3)
top_k=(3 5 10)
top_k_centrality=(3 5)
output_dim=(6 8 10)
hidden_dim=(12 16 24)
num_layers=(3 4 5)
pos_weight_exponent=(1 3 5 7 10 15)
dropout=(0.3 0.5 0.7)

# Random Search Parameters
NUM_TRIALS=50  # Adjust this to how many runs you have time for
best_acc=0
best_args=""

echo "Starting Random Search with $NUM_TRIALS trials..."

for (( i=1; i<=NUM_TRIALS; i++ )); do

    # Randomly select an element from each array
    lr=${learning_rate[$RANDOM % ${#learning_rate[@]}]}
    ws=${window_size[$RANDOM % ${#window_size[@]}]}
    tw=${temporal_window[$RANDOM % ${#temporal_window[@]}]}
    k=${top_k[$RANDOM % ${#top_k[@]}]}
    kc=${top_k_centrality[$RANDOM % ${#top_k_centrality[@]}]}
    o=${output_dim[$RANDOM % ${#output_dim[@]}]}
    hd=${hidden_dim[$RANDOM % ${#hidden_dim[@]}]}
    nl=${num_layers[$RANDOM % ${#num_layers[@]}]}
    pwe=${pos_weight_exponent[$RANDOM % ${#pos_weight_exponent[@]}]}
    dr=${dropout[$RANDOM % ${#dropout[@]}]}
    echo "================================================================="
    echo "Trial $i / $NUM_TRIALS"
    echo "Running: lr=$lr, ws=$ws, tw=$tw, top_k=$k, top_k_centrality=$kc, output_dim=$o, hidden_dim=$hd, num_layers=$nl, pos_weight_exp=$pwe, dropout=$dr"

    output=$(python3.12 run_GCTS.py \
        --learning_rate "$lr" \
        --window_size "$ws" \
        --max_epochs "100" \
        --horizon "$ws" \
        --temporal_window "$tw" \
        --output_dim "$o" \
        --hidden_dim "$hd" \
        --top_k "$k" \
        --top_k_centrality "$kc" \
        --num_layers "$nl" \
        --pos_weight_exponent "$pwe" \
        --experiment_name "LEED" \
        --dataset "telco" \
        --graph_type "corr" \
        --dropout "$dr" \
        2>&1)

    echo "$output"
    
    # Extract accuracy
    acc=$(echo "$output" | grep -oP 'test_accuracy:\s*\K[0-9]+(\.[0-9]+)?')

    # Safety check: ensure acc was actually found before doing math
    if [[ -n "$acc" ]]; then
        if (( $(echo "$acc > $best_acc" | bc -l) )); then
            best_acc=$acc
            best_args="lr=$lr ws=$ws tw=$tw top_k=$k top_k_centrality=$kc output_dim=$o hidden_dim=$hd num_layers=$nl pos_weight_exp=$pwe dropout=$dr"
            echo "*** NEW BEST ACCURACY: $best_acc ***"
        fi
    else
        echo "Warning: Could not extract test_accuracy from trial $i. The run may have crashed."
    fi

done

echo "==============================="
echo "RANDOM SEARCH COMPLETE"
echo "Best accuracy: $best_acc"
echo "Best args: $best_args"