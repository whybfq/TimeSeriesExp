for len in 96 192 336 720
do
  python -u /home/hangyi/Downloads/UniTS/TSLANet/Forecasting/TSLANet_Forecasting.py \
  --root_path /home/hangyi/Downloads/UniTS/dataset/ETT-small \
  --pred_len $len \
  --data ETTh2 \
  --data_path ETTh2.csv \
  --seq_len 512 \
  --emb_dim 64 \
  --depth 1 \
  --batch_size 512 \
  --dropout 0.5 \
  --patch_size 32 \
  --train_epochs 20 \
  --pretrain_epochs 10
done