#!/bin/bash
#this shell script help run the a2d2 training algorithm for video prediction
name="A2D2-alpha-3"
gpu_id='0,1'
batch_size=8
image_size=[240,152]
image_size_H=152
image_size_W=240
K=4
T=2
c_dim=3
model="STMF"
depth=22
growthRate=16
reduction=0.5
bottleneck=True
gf_dim=16
checkpoints_dir="./checkpoints"
tensorboard_dir="./tb"
txtroot="./data/"
data_root="./data"
lr=0.0001
nepoch=200
nepoch_decay=200
continue_train=False
which_epoch='latest'
alpha=3.0
beta1=0.5
adversarial=True
lr_policy='step'
lr_decay_iters=50
start_epoch=1
print_freq=10
display_freq=10
save_latest_freq=100
logDir = $tensorboard_dir/$name

tensorboard --logdir=./tb/$name --
python3 -m train_a2d2 --name $name --gpu_ids $gpu_id --batch_size $batch_size --image_size $image_size --image_size_H $image_size_H --image_size_W $image_size_W --K $K --T $T --c_dim $c_dim --model $model --depth $depth --growthRate $growthRate --reduction $reduction --bottleneck $bottleneck --gf_dim $gf_dim --checkpoints_dir $checkpoints_dir --tensorboard_dir $tensorboard_dir --txtroot $txtroot --data_root $data_root --lr $lr --nepoch $nepoch --nepoch_decay $nepoch_decay --continue_train $continue_train --which_epoch $which_epoch --alpha $alpha --beta1 $beta1 --lr_policy $lr_policy --lr_decay_iters $lr_decay_iters --start_epoch $start_epoch --print_freq $print_freq --display_freq $display_freq --save_latest_freq $save_latest_freq
