#!/bin/bash
#SBATCH --job-name=train_adversarial_bert_1
#SBATCH --output=test.out.%j
#SBATCH --error=vasp.err.%j
#SBATCH --partition=gpu4Q
#SBATCH --nodes=1
#SBATCH	--cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --qos=gpuq

source activate tf2.2
python adversarial_BERT.py