#!/bin/bash
#SBATCH --job-name=predict-roberta-base
#SBATCH --open-mode=append
#SBATCH --output=log/%j_%x.out
#SBATCH --error=log/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:0
#SBATCH --mem=25G
#SBATCH -c 4

singularity exec --nv --bind /scratch/tl2564 --overlay /scratch/tl2564/overlay-25GB-500K.ext3:ro /scratch/tl2564/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/tl2564/mllu-is-fre-good-indicator-for-dataset-complexity/experiment/roberta-base
python predict.py
"
