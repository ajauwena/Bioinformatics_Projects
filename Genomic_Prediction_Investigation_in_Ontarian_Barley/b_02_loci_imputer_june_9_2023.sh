#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=24:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Imputing_Loci_June_9_2023
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

module load beagle/5.4

dir="/home/ajauwena/scratch/binf_6999/barley_june_9_2023/analysis/tassel_barley_june_9_2023"

java -jar ${EBROOTBEAGLE}/beagle.22Jul22.46e.jar gt=$dir/barley_june_9_2023_platypus_loci_filtered_filtered.vcf out=$dir/barley_june_9_2023_platypus_loci_filtered_filtered_imputed