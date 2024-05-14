#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=30:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Trimming_Adapter_Sequences_2_June_9_2023
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# Cutadapt warned that the same repeated bases (e.g., "T," "G," etc.) often precede the adapter sequences in many reads. As such, I will trim five bases from the 5' end of both the forward and reverse reads.

source /home/ajauwena/cutadapt_env/bin/activate

module load python/3.10.2

dir_june_9_2023="/home/ajauwena/scratch/binf_6999/barley_june_9_2023"

parallel -j 16 cutadapt -u -5 -U -5 \
    --output $dir_june_9_2023/trimmed_twice/{}_trim_2x_forward.fastq.gz \
    --paired-output $dir_june_9_2023/trimmed_twice/{}_trim_2x_reverse.fastq.gz \
    $dir_june_9_2023/trimmed/{}_trim_forward.fastq.gz $dir_june_9_2023/trimmed/{}_trim_reverse.fastq.gz ::: $(ls -1 $dir_june_9_2023/trimmed/*.fastq.gz | xargs -n 1 basename | sed 's/_.*//' | sort -u)