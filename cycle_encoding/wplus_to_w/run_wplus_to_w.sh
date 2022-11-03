#!/bin/bash

python -u scripts/train.py \
	--dataset_type ffhq_encode \
	--exp_dir experiments/test \
	--start_from_latent_avg \
	--use_w_pool \
	--w_discriminator_lambda 0.1 \
	--id_lambda 0.1 \
	--lpips_lambda 0.8 \
	--inverse_progressive_start 650000 \
	--inverse_progressive_step_every 4000 \
	--delta_norm_lambda_progressive_step_every 10000 \
	--max_steps 750000 \
	--val_interval  10000 \
	--save_interval 5000 \
	--stylegan_size 1024 \
	--stylegan_weights pretrained_models/stylegan2-ffhq-config-f.pt \
	--workers 8 \
	--batch_size 8 \
	--test_batch_size 8 \
	--test_workers 8 \
	--save_training_data \
	--keep_optimizer \
	--resume_training_from_ckpt ../w_to_wplus/experiments/test/checkpoints/iteration_500000.pt 
