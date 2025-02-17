This Repo contain the sample codes for "Output Scaling: YingLong - Delayed Chain of Thought in a Large Pretrained Time Series Forecasting Model".

The dataset preprocessing code is in ```prepare_data.py```.

The training main file is ```pretrain/main.py```.

The model file is ```lit_gpt/model.py```.

A sample training script is
```
fabric run model \
    --accelerator=cuda \
    --devices=4 \
    --num-nodes=1 \
pretrain/main.py \
    --config_name tiny.yaml \
    --train_data_dir train_datasets \
    --val_data_dir val_datasets 
   ```
Here we consider the 4 GPU setting and the model configs is ```configs/tiny.yaml```.



To evaluate the model on ETTh/m and Weather datasets, a sample is
```
batch_size=32
scaling=1
model_name=tiny.yaml
ckp=100000
future_token=4096
seq_len=4096
num_gpus=4

fabric run model \
    --accelerator=cuda \
    --devices=$num_gpus \
    --num-nodes=1 \
eval_dist.py \
    --batch_size $batch_size \
    --seq_len $seq_len \
    --config_name $config_name \
    --ckp $ckp \
    --num_gpus $num_gpus \
    --future_token $future_token \
    -t ETTh1 -t ETTh2 -t ETTm1 -t ETTm2 -t Weather \
    -p 96 \
    --print \
```

