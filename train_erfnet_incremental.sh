#!/bin/bash

## If you are working with environments make sure you activated it before starting the script

# Stage 2, with first teacher
 python3 train_erfnet_incremental.py --model_name erfnet_incremental_set12 --train_set 2 --num_epochs 200 --teachers erfnet_static_set1 199 1 --validate
 python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch 199 --task_to_val 1
 python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch 199 --task_to_val 2
 python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch 199 --task_to_val 12

# Stage 3, with second teacher
 python3 train_erfnet_incremental.py --model_name erfnet_incremental_set123 --train_set 3 --num_epochs 200 --teachers erfnet_ctthesis_set12 199 2 --validate
 python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch 199 --task_to_val 1
 python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch 199 --task_to_val 2
 python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch 199 --task_to_val 3
 python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch 199 --task_to_val 12
 python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch 199 --task_to_val 123


echo "Completed job on "$(hostname)
