#!/bin/bash

#SBATCH --job-name=comp_gen_cogs_train_gru_uni_with_att_hidden512_no_if_run2
#SBATCH --open-mode=append
#SBATCH --output=/scratch/str8775/conda_env/compositional-generalisation/COGS/exp/run2_2layers_gru_512/%x_%j.out
#SBATCH --error=/scratch/str8775/conda_env/compositional-generalisation/COGS/exp/run2_2layers_gru_512/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

singularity exec --nv --bind /scratch --overlay /scratch/str8775/conda_env/compositional-generalisation_env.ext3:ro /scratch/str8775/conda_env/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "
source /ext3/env.sh
conda activate
cd /scratch/str8775/conda_env/compositional-generalisation/COGS/scripts/
bash run_gru_uni.sh ../exp_data ../src/OpenNMT-py
"
