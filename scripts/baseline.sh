#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=unlearning
#SBATCH --output=baseline.out

module purge

singularity exec --nv \
	    --overlay /scratch/sg7729/my_env/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python /scratch/sg7729/machine-unlearning/unlearn_harm.py --model_name=facebook/opt-1.3b --model_save_dir=/scratch/sg7729/machine-unlearning/models/opt1.3b_unlearned --log_file=/scratch/sg7729/machine-unlearning/logs/opt-1.3b-unlearn.log"

		