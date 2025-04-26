#!/bin/bash
#SBATCH --job-name=dbscan_sweep
#SBATCH --output=logs/dbscan_sweep_%A_%a.out
#SBATCH --error=logs/dbscan_sweep_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.zhang.ddz5@yale.edu
#SBATCH --partition bigmem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=01:00:00
#SBATCH --array=0-5

# Define parameter grids
epsilons=(0.17 0.20)
minpts=(5 10 15)

# Compute index from SLURM_ARRAY_TASK_ID
eps_idx=$(( SLURM_ARRAY_TASK_ID / 3 ))
minpts_idx=$(( SLURM_ARRAY_TASK_ID % 3 ))

eps=${epsilons[$eps_idx]}
min=${minpts[$minpts_idx]}

input="/home/sr2464/Desktop/cpsc524FinalProject/example_data/rnaseq_sample5000d_10000_cells.csv"
output="/home/ddz5/Desktop/cpsc524FinalProject/example_data/rnaseq_sample5000d_10000_cells_eps_${eps}_minpts_${min}_clustered.csv"

echo "Running DBSCAN with eps=$eps, minPts=$min"
../dbscan_kdtree -i "$input" --no-trees --eps "$eps" --min-pts "$min" -o "$output"
