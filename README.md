# MRI-Hippocampus-Segmentation
# Train model
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --do_predict \
    --batch_size 16 \
    --epochs 100 \
    --train_dir YOUR DATA DIRECTORY \
    --eval_dir YOUR DATA DIRECTORY \
    --seg_dir YOUR DATA DIRECTORY \
    --output_dir YOUR OUTPUT DIRECTORY \
    
# Evaluate model
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_predict \
    --eval_dir YOUR DATA DIRECTORY \
    --seg_dir YOUR DATA DIRECTORY \
    --output_dir YOUR OUTPUT DIRECTORY \
    --checkpoint YOUR CHECKPOINT FILE PATH \

