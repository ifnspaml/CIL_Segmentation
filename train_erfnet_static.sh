#!/bin/bash

# Stage 1, first Teacher
 python3 train_erfnet_static.py --model_name erfnet_incremental_set1 --train_set 1 --num_epochs 200 --city --validate
 python3 evaluate/evaluate_erfnet.py --load_model_name erfnet_incremental_set1 --train_set 1 --weights_epoch 199 --task_to_val 1

# All classes baseline
 python3 train_erfnet_static.py --model_name erfnet_static_set123 --train_set 123 --num_epochs 200 --validate
 python3 evaluate/evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch 199 --task_to_val 1
 python3 evaluate/evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch 199 --task_to_val 2
 python3 evaluate/evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch 199 --task_to_val 12
 python3 evaluate/evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch 199 --task_to_val 3
 python3 evaluate/evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch 199 --task_to_val 123


echo "Completed job on "$(hostname)
