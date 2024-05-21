#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=24:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Renaming_Samples_in_VCF_File_June_9_2023
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

module load StdEnv/2020 gcc/9.3.0 bcftools/1.16

dir="/home/ajauwena/scratch/binf_6999/barley_june_9_2023/analysis/tassel_barley_june_9_2023"

bcftools reheader -s $dir/sample_names_june_9_2023.txt -o $dir/barley_june_9_2023_platypus_loci_filtered_filtered_imputed_filtered_renamed_for_merging.vcf $dir/barley_june_9_2023_platypus_loci_filtered_filtered_imputed_filtered.vcf