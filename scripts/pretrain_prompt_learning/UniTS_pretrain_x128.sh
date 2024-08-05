model_name=UniTS
exp_name=UniTS_pretrain_x128
wandb_mode=online
ptune_name=prompt_tuning

d_model=128

random_port=$((RANDOM % 9000 + 1000))

# Pretrain # 总迭代次数 = train_epochs * (N / batch_size)
# 例如，如果你的数据集有3200个样本，batch size为32，那么每个epoch的迭代次数是 3200 / 32 = 100。如果你训练10个epoch，那么总迭代次数就是 10 * 100 = 1000。
torchrun --nnodes 1 --nproc-per-node 1 --master_port $random_port /home/hangyi/Downloads/UniTS/run_pretrain.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model $d_model \
  --des 'Exp' \
  --acc_it 128 \
  --batch_size 32 \
  --learning_rate 5e-5 \
  --min_lr 1e-4 \
  --weight_decay 5e-6 \
  --train_epochs 3 \
  --warmup_epochs 0 \
  --min_keep_ratio 0.5 \
  --right_prob 0.5 \
  --min_mask_ratio 0.7 \
  --max_mask_ratio 0.8 \
  --debug $wandb_mode \
  --task_data_config_path data_provider/multi_task_pretrain.yaml

# Prompt tuning
torchrun --nnodes 1 --master_port $random_port /home/hangyi/Downloads/UniTS/run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --lradj prompt_tuning \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 3e-3 \
  --weight_decay 0 \
  --prompt_tune_epoch 2 \
  --train_epochs 0 \
  --acc_it 32 \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --clip_grad 100 \
  --pretrained_weight auto \
  --task_data_config_path  data_provider/multi_task.yaml
