#!/usr/bin/env bash

gpus=0

data_name=SYSU
net_G=FreqFormer #This is the best version
split=test
vis_root=/mnt/data1/gyf/code/newtest/ChangeFormer/vis6/SYSU_FreqFormer
project_name=SYSU_FreqFormer
checkpoints_root=/mnt/data1/gyf/code/newtest/ChangeFormer/checkpoint4/SYSU1/CD_FreqFormer_SYSU_b16_lr0.0002_adamw_train_val_300_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256
checkpoint_name=/mnt/data1/gyf/code/newtest/ChangeFormer/checkpoint4/SYSU1/CD_FreqFormer_SYSU_b16_lr0.0002_adamw_train_val_300_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/best_ckpt.pt
img_size=256
embed_dim=256 #Make sure to change the embedding dim (best and default = 256)

CUDA_VISIBLE_DEVICES=0 python eval_cd.py --split ${split} --net_G ${net_G} --embed_dim ${embed_dim} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


