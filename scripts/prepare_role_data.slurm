#!/bin/bash

#SBATCH --job-name=comp_gen_cogs_train_gru_uni_att_hidden512_no_if_new_vocab_seed1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/str8775/conda_env/compositional-generalisation/COGS/exp/run1/seed100_%x_%j.out
#SBATCH --error=/scratch/str8775/conda_env/compositional-generalisation/COGS/exp/run1/seed100_%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

export ENV=$1
export SEED=$2

echo singularity exec --nv --bind /scratch --overlay /scratch/str8775/conda_env/compositional-generalisation_env${ENV}.ext3:ro /scratch/str8775/conda_env/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "
source /ext3/env.sh
conda activate
cd /scratch/str8775/conda_env/compositional-generalisation/COGS/scripts/
bash run_prepare_role_data.sh ../exp_data ../src/OpenNMT-py ${SEED}
"


singularity exec --nv --bind /scratch --overlay /scratch/str8775/conda_env/compositional-generalisation_env${ENV}.ext3:rw /scratch/str8775/conda_env/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "
source /ext3/env.sh
conda activate
cd /scratch/str8775/conda_env/compositional-generalisation/COGS/scripts/
bash run_prepare_role_data.sh ../exp_data ../src/OpenNMT-py ${SEED}
"

