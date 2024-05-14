#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=04:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Using_FastQC_2_June_9_2023
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

module load fastqc/0.11.9

fastqc /home/ajauwena/scratch/binf_6999/barley_june_9_2023/trimmed_twice/*.fastq.gz --outdir /home/ajauwena/scratch/binf_6999/barley_june_9_2023/analysis/fastqc_barley_june_9_2023_trimmed --threads 16