for len in 96 192 336 720
do
  python -u /home/hangyi/Downloads/UniTS/TSLANet/Forecasting/TSLANet_Forecasting.py \
  --root_path /home/hangyi/Downloads/UniTS/dataset/electricity \
  --pred_len $len \
  --data custom \
  --data_path electricity.csv \
  --seq_len 512 \
  --emb_dim 128 \
  --depth 2 \
  --batch_size 16 \
  --dropout 0.5 \
  --patch_size 32 \
  --train_epochs 10 \
  --pretrain_epochs 10
done