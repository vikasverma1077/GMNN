#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=4                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=3:00:00                   # The job will run for 3 hours
#SBATCH -o slurm-%j.out  # Write the log in $SCRATCH

# 1. Create your environement locally
module load python/3.6
#virtualenv --no-download $SLURM_TMPDIR/env
source /home/vermavik/virtualenv/al/bin/activate
#pip install --no-index torch torchvision



# 2. Copy your dataset on the compute node
#cp $SCRATCH/<dataset> $SLURM_TMPDIR

python train.py --dataset ../data/pubmed  --mixup_alpha 1.0 --mixup_consistency 1.0 --seed 1
python train.py --dataset ../data/pubmed  --mixup_alpha 1.0 --mixup_consistency 1.0 --seed 2
python train.py --dataset ../data/pubmed  --mixup_alpha 1.0 --mixup_consistency 1.0 --seed 3
python train.py --dataset ../data/pubmed  --mixup_alpha 1.0 --mixup_consistency 1.0 --seed 4
