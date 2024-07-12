#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=24:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Filtering_and_Imputing_Loci_June_9_2023
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# NOTE: I manually removed the window "CHROM=ENA|CAJHDD010000032|CAJHDD010000032.1, POS=9298" because it only has one position, which prevented Beagle from calculating the appropriate statistical measures needed for imputation.

module load vcftools/0.1.16

module load StdEnv/2020

module load beagle/5.4

dir="/home/ajauwena/scratch/binf_6999/barley_june_9_2023/analysis/tassel_barley_june_9_2023"

vcftools --vcf $dir/barley_june_9_2023_platypus_snp_filtered.vcf --max-missing 0.2 --recode --recode-INFO-all --out $dir/barley_june_9_2023_platypus_loci_filtered

java -jar ${EBROOTBEAGLE}/beagle.22Jul22.46e.jar gt=$dir/barley_june_9_2023_platypus_loci_filtered.recode.vcf out=$dir/barley_june_9_2023_platypus_loci_filtered_imputed