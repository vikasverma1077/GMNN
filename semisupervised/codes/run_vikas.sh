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

#python train.py --dataset ../data_subset/pubmed/n5v5/1 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0
#python train.py --dataset ../data_subset/pubmed/n5v5/2 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0
#python train.py --dataset ../data_subset/pubmed/n5v5/3 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0
#python train.py --dataset ../data_subset/pubmed/n5v5/4 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0
#python train.py --dataset ../data_subset/pubmed/n5v5/5 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0



#python train.py --dataset ../data_subset/pubmed/n10v10/1 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0
#python train.py --dataset ../data_subset/pubmed/n10v10/2 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0
#python train.py --dataset ../data_subset/pubmed/n10v10/3 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0
#python train.py --dataset ../data_subset/pubmed/n10v10/4 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0
#python train.py --dataset ../data_subset/pubmed/n10v10/5 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0

#python train.py --dataset ../data_subset/pubmed/n20/1 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0
#python train.py --dataset ../data_subset/pubmed/n20/2 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0
#python train.py --dataset ../data_subset/pubmed/n20/3 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0
python train.py --dataset ../data_subset/pubmed/n20/4 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0
python train.py --dataset ../data_subset/pubmed/n20/5 --only_gnn --do_range 0.5 --mixup_alpha_range 0.0 --consis_coeff_range 0.0

