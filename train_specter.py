#!/usr/bin/env python
import sys
# Import to register the dataset
import preprocess_triplets  # This registers 'specter_triplets'

# Now run the training
from swift.llm import sft_main

# Set up command line arguments as if running from CLI
sys.argv = [
    'train_specter.py',
    '--model', 'Qwen/Qwen3-Embedding-0.6B',
    '--task_type', 'embedding',
    '--model_type', 'qwen3_emb',
    '--train_type', 'full',
    '--dataset', 'specter_triplets',
    '--output_dir', 'output/specter_model/', #/mount/weka/shriya/qwen_training/swift',
    '--loss_type', 'infonce',
    '--num_train_epochs', '1',
    '--logging_steps', '10',
    '--save_steps', '50',
    '--learning_rate', '6e-6',
    '--max_length', '512',  # Qwen3-Embedding supports up to 32768
    '--truncation_strategy', 'right',  # Keep title + start of abstract
    '--logging_dir', 'output/logs',  # For TensorBoard,
    '--torch_dtype', 'bfloat16',
    '--max_epochs', '1',
    '--per_device_train_batch_size', '8',
    '--gradient_accumulation_steps', '8',
    '--max_steps', '200',
    '--ddp_backend', 'nccl',
    '--eval_steps', '50',  # Evaluate every 50 steps
    '--evaluation_strategy', 'steps',  # Or 'epoch'
]

sft_main()
