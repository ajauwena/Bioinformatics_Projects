#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=24:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Demultiplexing_May_2022
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

module load sabre

sabre pe -f /home/ajauwena/scratch/binf_6999/barley_may_2022/reads/NS.1892.004.D707---B503.Lukens_20220503_Plate1_R1.fastq.gz \
-r /home/ajauwena/scratch/binf_6999/barley_may_2022/reads/NS.1892.004.D707---B503.Lukens_20220503_Plate1_R2.fastq.gz \
-b /home/ajauwena/scratch/binf_6999/barley_may_2022/barcodes/Gbs_Illumina_PstI-PlatePositions_May_2022_Modified.txt \
-u /home/ajauwena/scratch/binf_6999/barley_may_2022/demultiplexing/unknown_forward.fastq \
-w /home/ajauwena/scratch/binf_6999/barley_may_2022/demultiplexing/unknown_reverse.fastq \

gzip /home/ajauwena/scratch/binf_6999/barley_may_2022/demultiplexing/*.fastq