#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=30:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Trimming_Adapter_Sequences_1_May_2022
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

source /home/ajauwena/cutadapt_env/bin/activate

module load python/3.10.2

dir_may_2022="/home/ajauwena/scratch/binf_6999/barley_may_2022"

parallel -j 16 cutadapt -u 10 -U 10 -q 15 -m 50 \
    -a AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC -A AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT \
    --output $dir_may_2022/trimmed/{}_trim_forward.fastq.gz \
    --paired-output $dir_may_2022/trimmed/{}_trim_reverse.fastq.gz \
    $dir_may_2022/demultiplexing/{}_forward.fastq.gz $dir_may_2022/demultiplexing/{}_reverse.fastq.gz ::: $(ls -1 $dir_may_2022/demultiplexing/*.fastq.gz | xargs -n 1 basename | sed 's/_.*//' | sort -u)