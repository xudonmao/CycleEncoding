#!/bin/bash


python -u scripts/refine.py \
       --dataset_type ffhq_encode \
       --exp_dir experiments/test \
       --start_from_latent_avg \
       --use_w_pool \
       --learning_rate 0.0001 \
       --id_lambda 0.1 \
       --lpips_lambda 0.8 \
       --max_steps 15 \
       --val_interval  1000 \
       --save_interval 5000 \
       --stylegan_size 1024 \
       --stylegan_weights pretrained_models/stylegan2-ffhq-config-f.pt \
       --workers 8 \
       --batch_size 7 \
       --test_batch_size 1 \
       --test_workers 2 \
       --save_training_data \
       --keep_optimizer \
       --resume_training_from_ckpt pretrained_models/cycle_encoding_ffhq_encode.pt

