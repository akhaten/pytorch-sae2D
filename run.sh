#!/bin/bash

#SBATCH --job-name=Seg
#SBATCH --output=trains/train_osirim/output.out
#SBATCH --error=trains/train_osirim/error.err

#SBATCH --mail-type=END
#SBATCH --mail-user=Jessy.Khafif@irit.fr

#SBATCH --partition=GPUNodes

#SBATCH --nodelist=gpu-nc07

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding


container=/projets/minds/jkhafif/containers/pytorch-with-ignite-v1.sif
directory=/users/minds/jkhafif/Documents/pytorch-sae2D
python=python3
# script=run.py
script=run.py


srun singularity exec ${container} ${python} ${script} ${directory}/trains/train_osirim