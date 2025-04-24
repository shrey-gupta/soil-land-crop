#!/bin/bash
#SBATCH --job-name=Corn-lstm-detrend 		# Job name
#SBATCH --time=3-00:00:00  			# 3 days
#SBATCH --error=error_lstm_detrend.err 		# Error file
#SBATCH --output=output_lstm_detrend.out  	# Output file
#SBATCH --nodes=1 				# How many nodes to run on
#SBATCH --ntasks=1 --cpus-per-task=4 		# 4 cpu on single node
#SBATCH --partition=long 			# Partition CPU node to run your job on
#SBATCH --gres=gpu:4                     	# Number of GPUs per node
#SBATCH --mem=48G 				# Job memory request
#SBATCH --mail-type=BEGIN,END,FAIL 		# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=guptsh@bc.edu 		# Where to send mail

source ~/shrey-virtual-hpc/bin/activate

cd /home/guptsh/working-directory/crop-yield/py-models/

python lstm-no-attn-detrend-regression.py
