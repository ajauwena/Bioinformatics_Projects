#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=24:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Demultiplexing_June_9_2023
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

module load sabre

sabre pe -f /home/ajauwena/scratch/binf_6999/barley_june_9_2023/reads/NS.2145.004.B716---B501.Lukens-P-TC1_R1.fastq.gz \
-r /home/ajauwena/scratch/binf_6999/barley_june_9_2023/reads/NS.2145.004.B716---B501.Lukens-P-TC1_R2.fastq.gz \
-b /home/ajauwena/scratch/binf_6999/barley_june_9_2023/barcodes/Gbs_Illumina_PstI-PlatePositions_June_9_2023_Modified.txt \
-u /home/ajauwena/scratch/binf_6999/barley_june_9_2023/demultiplexing/unknown_forward.fastq \
-w /home/ajauwena/scratch/binf_6999/barley_june_9_2023/demultiplexing/unknown_reverse.fastq \

gzip /home/ajauwena/scratch/binf_6999/barley_june_9_2023/demultiplexing/*.fastq