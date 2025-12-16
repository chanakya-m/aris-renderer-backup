#!/bin/zsh

shapes=("armadillo" "bunny" "cow" "dragon" "teapot")
counts=(10000 50000 100000)

resolution=128
batch_size=32786

for shape in $shapes; do
    for count in $counts; do

        ckpt="checkpoints/igr_final_${shape}_${count}_int8.pth"

        out="data/${shape}/${shape}_reconstructed_${count}_int8.ply"

        echo "========================================================"
        echo "Processing: ${shape} | Points: ${count} | INT8"

        if [[ -f "$ckpt" ]]; then
            python reconstruct_igr_qat.py \
                --checkpoint "$ckpt" \
                --output "$out" \
                --resolution $resolution \
                --batch_size $batch_size
        else
            echo "Skipping: Checkpoint not found at $ckpt"
        fi

    done
done
