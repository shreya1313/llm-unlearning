#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=unlearning
#SBATCH --output=/scratch/sg7729/machine-unlearning/logs/eval_2.out

module purge

singularity exec --nv \
	    --overlay /scratch/sg7729/my_env/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python /scratch/sg7729/machine-unlearning/evaluate_dataset_opt2.py"

		
