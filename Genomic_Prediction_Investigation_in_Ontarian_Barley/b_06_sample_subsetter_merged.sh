#!/bin/bash
#SBATCH --account=def-lukens
#SBATCH --time=24:00:00
#SBATCH --mail-user=ajauwena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=Subsetting_Samples_in_Merged_VCF
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

module load StdEnv/2020 gcc/9.3.0 bcftools/1.16

dir="/home/ajauwena/scratch/binf_6999/barley_merged"

bcftools view -S $dir/overlapping_varieties.txt -O v -o $dir/barley_merged_platypus_loci_imputed_subsetted.vcf $dir/barley_merged_platypus_loci_imputed.vcf.gz