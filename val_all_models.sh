#!/bin/bash

## Incremental

# Stage 1
python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set1 --train_set 1 --weights_epoch 199 --task_to_val 1

# Stage 2
python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch 199 --task_to_val 1
python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch 199 --task_to_val 2
python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch 199 --task_to_val 12

# Stage 3
python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch 199 --task_to_val 1
python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch 199 --task_to_val 2
python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch 199 --task_to_val 3
python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch 199 --task_to_val 12
python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch 199 --task_to_val 123


## Static

# All classes
python3 evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch 199 --task_to_val 1
python3 evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch 199 --task_to_val 2
python3 evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch 199 --task_to_val 12
python3 evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch 199 --task_to_val 3
python3 evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch 199 --task_to_val 123


